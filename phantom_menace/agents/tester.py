import contextlib
from dataclasses import dataclass
import multiprocessing
import io
from collections import defaultdict
import traceback

import pandas as pd
from pydantic import BaseModel, Field
import sklearn
import numpy as np
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIResponsesModel,
    ModelSettings,
    function_tool,
)
from agents.memory.sqlite_session import SQLiteSession
import shap
import statsmodels as sm
import scipy as spy
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error

from rogueone.utils import llm_cfg, test_agent_cfg
from rogueone.utils.console import ConsoleManager
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.wandb_utils import wandb_logging_wrapper
from rogueone.llm.prompts.agent_roles import AgentRoles
from rogueone.llm.agents.web_search import WebSearchAgent
from brainflux.external_connectors.brainflux_filter_pipeline import (
    BrainfluxFilterPipeline,
)

# set_tracing_disabled(True)

GPUS = ["cuda:0"]


@dataclass
class TestResultClassification:
    global_recall: float
    loss: float | None = None
    best_threshold: float | None = None
    target_class_patients_removed: int | None = None
    non_target_class_patients_removed: int | None = None
    top_impactful_attributes: dict[str, float] | None = None
    report: str | None = None

    def to_dict(self, only_metrics: bool = True) -> dict:
        if only_metrics:
            tmp = self.to_dict(only_metrics=False)
            tmp.pop("report", None)
            return tmp
        else:
            tmp = {
                "global_recall": self.global_recall,
                "loss": self.loss,
                "best_threshold": self.best_threshold,
                "target_class_patients_removed": self.target_class_patients_removed,
                "non_target_class_patients_removed": self.non_target_class_patients_removed,
                "report": self.report,
            }
            for k, v in (self.top_impactful_attributes or {}).items():
                tmp[f"top_impactful_attribute_{k}"] = v
            return tmp

    @property
    def as_dict(self) -> dict:
        return self.to_dict(only_metrics=False)


class CodeInput(BaseModel):
    code: str = Field(
        ..., description="Mandatory. The python code to execute.", min_length=1
    )
    reasoning: str = Field(
        ...,
        description="Mandatory. The reasoning behind the code execution. Explain why this code is being executed and what it aims to achieve.",
        min_length=1,
    )
    pseudo_code: str = Field(
        ...,
        description="Mandatory. Pseudo code representation of the code to execute. This is to help with understanding the logic before execution and fixing potential issues.",
        min_length=1,
    )


class TesterAgent:

    def __init__(
        self,
        cfg: ExperimentConfig,
        precision_min: float,
        tm_frac: float,
        target_class: int = 0,
    ):
        self.cfg_experiment = cfg
        self.knowledge_agent = WebSearchAgent(
            cfg=self.cfg_experiment  # , collection_name="tester_knowledge_collection"
        )
        self.target_class = target_class
        self.precision_min = precision_min
        self.tm_frac = tm_frac

        self._brainflux_filter_pipeline = BrainfluxFilterPipeline(
            data_path="/workspace/phantom_menace/Suppression_Ratio",
            label_path_train="/workspace/phantom_menace/train.csv",
            label_path_test="/workspace/phantom_menace/test.csv",
            target_class=self.target_class,
            min_precision=self.precision_min,
        )

        # self._brainflux_filter_pipeline_test = BrainfluxFilterPipeline(
        #     data_path="/workspace/phantom_menace/Suppression_Ratio",
        #     label_path_train="/workspace/phantom_menace/combined.csv",
        #     label_path="/workspace/phantom_menace/test.csv",
        #     target_class=self.target_class,
        #     min_precision=self.precision_min,
        # )

        self._client = AsyncOpenAI(
            base_url=llm_cfg.endpoint,
            api_key=llm_cfg.api_key,
            organization="brainflux-inc",
            project="brainflux_project",
            webhook_secret=None,
        )

    @staticmethod
    def prune_attributes_in_df(
        df: pd.DataFrame, df_attribute_explanations: pd.DataFrame
    ) -> pd.DataFrame:
        pruned_attributes = df_attribute_explanations[
            df_attribute_explanations["Status"] == "Pruned"
        ]["Attribute"].tolist()

        df_pruned = df.drop(columns=pruned_attributes, errors="ignore")

        return df_pruned

    async def test_hypotheses(
        self,
        df_attributes_folds: list[pd.DataFrame],
        df_attribute_explanations: pd.DataFrame,
        *,
        step: int | None = None,
    ) -> tuple[TestResultClassification, pd.DataFrame]:

        df_attributes = pd.concat(
            [df_attributes_folds[j] for j in range(len(df_attributes_folds))],
            ignore_index=True,
        )

        df_attributes = self.prune_attributes_in_df(
            df_attributes, df_attribute_explanations
        )

        df_attributes_train = df_attributes[
            df_attributes["id"].isin(
                self._brainflux_filter_pipeline.all_patient_ids_train
            )
        ]
        df_attributes_test = df_attributes[
            df_attributes["id"].isin(
                self._brainflux_filter_pipeline.all_patient_ids_test
            )
        ]

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._num_features_pruned = 0

        _note_book = ""

        X_all_train = df_attributes_train.copy().drop(columns=["id"])
        X_all_test = df_attributes_test.copy().drop(columns=["id"])

        if X_all_train.shape[1] == 0:
            ConsoleManager.console_error_print("No feature columns -- skipping XGBoost, global_recall=0.0")
            return TestResultClassification(global_recall=0.0), df_attribute_explanations

        trouble_makers = self._brainflux_filter_pipeline.get_troublemaker(
            excluded_patients=[], top_p=self.tm_frac, set="train"
        )

        trouble_makers = {k: v for k, v in trouble_makers.items()}

        y_all_train = df_attributes_train["id"].map(trouble_makers).fillna(0)
        y_all_train = pd.DataFrame(
            {"id": df_attributes_train["id"], "target": y_all_train}
        )

        y_all_test = df_attributes_test["id"].map(trouble_makers).fillna(0)
        y_all_test = pd.DataFrame(
            {"id": df_attributes_test["id"], "target": y_all_test}
        )

        @function_tool
        async def search_in_literature_tool(question: str) -> str:
            """
            Ask the Knowledge Agent to look up questions related to feature testing in from the literature.
            This includes general questions about what methodology exists for performing certain types of tests.
            And also specific questions about how certain types of features have been evaluated in the literature.
            And how to compare different methodologies for evaluating features.

            Args:
                question (str): The question to look up.

            Returns:
                str: An answer to the question rooted in the literature.
            """

            try:
                self._tool_call_counter["search_in_literature_tool"] += 1
                res = await self.knowledge_agent.explain_query(question)

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "methodology_lookup_tool",
                    f"Question: {question}\nAnswer: {res[:500]}...",
                )

                self._tool_call_counter_success["search_in_literature_tool"] += 1
                return res
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error looking up methodology: {e}"
                )

        @function_tool
        async def take_note_tool(note: str) -> str:
            """Use this tool to note down important discoveries and facts that should be included in the final rapport.
            The notes can be retrieved later using the 'get_notes_tool'.

            Args:
                note (str): Note to take.

            Returns:
                str: Confirmation message.
            """
            try:
                self._tool_call_counter["take_note_tool"] += 1
                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "take_note_tool",
                    f"Taking note: {note}",
                )

                nonlocal _note_book
                _note_book += f"{note}\n"

                self._tool_call_counter_success["take_note_tool"] += 1
                return "Note added successfully."
            except Exception as e:
                return ConsoleManager.console_error_print(f"Error taking note: {e}")

        @function_tool
        async def get_notes_tool() -> str:
            """Use this tool to retrieve all notes taken so far using the 'take_note_tool'.


            Returns:
                str: All notes taken so far.

            """

            try:
                self._tool_call_counter["get_notes_tool"] += 1
                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "get_notes_tool",
                    "Getting all notes",
                )

                self._tool_call_counter_success["get_notes_tool"] += 1
                return _note_book
            except Exception as e:
                return ConsoleManager.console_error_print(f"Error getting notes: {e}")

        @function_tool
        async def generic_python_executor_tool(code_input: CodeInput) -> str:
            """This tool is a general python execution tool. It can be used to set up experiments to assess the usefulness of attributes.
            The code provided should be self-contained and include all necessary imports.
            The code will be passed on to an code execution agent that will run the code and return the output.
            Use this tool to run experiments that assess the predictive power of the attributes with respect to the target variable.

            Args:
                code (str): The python code to execute.
                reasoning (str): The reasoning behind the code execution. Explain why this code is being executed and what it aims to achieve.
                pseudo_code (str): Pseudo code representation of the code to execute. This helps the code execution agent to understand the logic before execution and fix potential issues.

            The environment has access to a attribute data (df_attributes) as pandas DataFrames, and target variable (y) as a numpy array.
            All code is non persistent and stateless between calls.

            The "test_pipeline" variable contains the BrainfluxFilterPipeline instance that can be used to get scores and troublemakers.
            The test_pipeline has the following methods:
                - get_score(excluded_patients: list[str]) -> float:
                    Args:
                        excluded_patients (list[str]): List of patient IDs to exclude from the score calculation.
                    Returns:
                        float: The global recall score after excluding the given patients.
                    Use this to assess the overall predictive power of the attributes by creating a classifier that predicts which patients should be excluded.
                    By excluding troublesome patients, you can evaluate how well the attributes identify problematic cases.
                    The higher the score, the better the attributes are at predicting the target variable.
                - get_troublemaker(excluded_patients: list[str], top_k: int) -> dict[str, float]:
                    Args:
                        excluded_patients (list[str]): List of patient IDs to exclude from the troublemaker calculation.
                        top_k (int): The number of top troublemakers to return.
                    Returns:
                        dict[str, float]: The patient IDs and their corresponding troublemaker scores for the top_k patients.
                    The patient IDs returned are the ones that are the are blocking further improvements in the predictive power of the attributes.
                    Use this to identify patients that are not classified well by the attributes.


            The environment have the following libraries imported and available for use:
            - pandas as pd
            - numpy as np
            - sklearn
            - shap
            - statsmodels as sm
            - scipy as spy
            - xgboost as xgb

            The dataframe 'df_attributes' has the following structure:
            - Each row represents an instance.
            - Each column represents a feature/attribute.

            The variable 'y' is a pandas dataframe that contains the target variable for each patient ID in 'df_attributes'.
            - example: y = pd.DataFrame({"id": [...], "target": [...]}) where "target" indicates the class label for each patient.

            The output of the code should be assigned to a variable named 'output'.

            Returns:
                str: The output of the executed code or an error message if execution fails.
            """

            global g_code
            g_code = None

            @function_tool
            def execute_code_tool(code: str) -> str:
                try:

                    assert (
                        "import os" not in code
                    ), "Importing os module is not allowed."
                    assert (
                        "import sys" not in code
                    ), "Importing sys module is not allowed."
                    self._tool_call_counter["generic_python_executor_tool"] += 1

                    with contextlib.redirect_stdout(io.StringIO()) as f:
                        local_vars = {}
                        exec(
                            code,
                            {
                                "y": y_all_train,
                                "df_attributes": df_attributes_train,
                                "pd": pd,
                                "np": np,
                                "sklearn": sklearn,
                                "shap": shap,
                                "sm": sm,
                                "spy": spy,
                                "xgb": xgb,
                                "test_pipeline": self._brainflux_filter_pipeline,
                            },
                            local_vars,
                        )

                        output = local_vars.get("output", "No output variable defined.")

                    output = "\n".join(f.readlines()) + f"{output}"

                    global g_code
                    g_code = code
                except Exception as e:
                    output = f"Error executing the code: {e}"  # \n{traceback.format_exc(limit=1)}
                    ConsoleManager.console_error_print(message=output, code=code)

                return output

            agent = Agent(
                model_settings=ModelSettings(
                    temperature=test_agent_cfg.temperature,
                    reasoning=test_agent_cfg.reasoning,
                ),
                name="Python Code Executor Agent",
                instructions="""
                    You are a python code execution agent.
                    You have access to a tool called 'execute_code_tool' that allows you to execute python.
                    Use this tool to execute the provided code and return the output.
                    If there are any errors in the code, fix them and try again.
                    Take time and think step-by-step before executing the code.
                    Return only the output of the code execution.
                    
                    The environment have the following libraries imported and available for use:
                    - pandas as pd
                    - numpy as np
                    - sklearn
                    - shap
                    - statsmodels as sm
                    - scipy as spy
                    - xgboost as xgb
                    
                    There is a pandas dataframe called 'df_attributes' that contains the attributes to be tested.
                    - Each row represents an instance.
                    - Each column represents a feature/attribute.
                    'df_attributes' is already loaded and available for use.                    

                    The variable 'y' is a pandas dataframe that contains the target variable for each patient ID in 'df_attributes'.
                    - example: y = pd.DataFrame({"id": [...], "target": [...]}) where "target" indicates the class label for each patient.
                    
                    There is a tester agent that is using you to execute python code to set up experiments to assess the usefulness of attributes.
                    The tester agent will provide you with the code to execute, the reasoning behind the code, and the pseudo code representation of the code.
                    Use this information to understand the context and purpose of the code execution.
                    
                    The "test_pipeline" variable contains the BrainfluxFilterPipeline instance that can be used to get scores and troublemakers.
                    The test_pipeline has the following methods:
                        - get_score(excluded_patients: list[str]) -> float:
                            Args:
                                excluded_patients (list[str]): List of patient IDs to exclude from the score calculation.
                            Returns:
                                float: The global recall score after excluding the given patients.
                            Use this to assess the overall predictive power of the attributes by creating a classifier that predicts which patients should be excluded.
                            By excluding troublesome patients, you can evaluate how well the attributes identify problematic cases.
                            The higher the score, the better the attributes are at predicting the target variable.
                        - get_troublemaker(excluded_patients: list[str], top_k: int) -> dict[str, float]:
                            Args:
                                excluded_patients (list[str]): List of patient IDs to exclude from the troublemaker calculation.
                                top_k (int): The number of top troublemakers to return.
                            Returns:
                                dict[str, float]: The patient IDs and their corresponding troublemaker scores for the top_k patients.
                            The patient IDs returned are the ones that are the are blocking further improvements in the predictive power of the attributes.
                            Use this to identify patients that are not classified well by the attributes.
                    
                    ## Important Guidelines:
                    - Always think step-by-step and explain your reasoning.
                    - Return only the output of the code execution.
                    - Do not load any data or files. All the needed data is already loaded and available for use in the 'df_attributes' pandas DataFrame and 'y' numpy array.
                    - Do not import os or sys modules.
                    - Do not create new features or modify the 'df_attributes' dataframe. Only use it as is for testing.
                    
                """,
                model=OpenAIResponsesModel(
                    model=llm_cfg.model,
                    openai_client=self._client,
                ),
                tools=[execute_code_tool],
            )

            output = await Runner.run(
                agent,
                f"""
                    You are to execute the following python code:
                    {code_input.code}  
                    The reasoning behind this code execution is as follows:
                    {code_input.reasoning}
                    The pseudo code representation of the code is as follows:
                    {code_input.pseudo_code or "N/A"}
                    Execute the code and provide the output.
                """,
                max_turns=20,
            )

            output = output.final_output

            max_output_length = min(len(output), 500)

            ConsoleManager.console_agent_logging(
                "TesterAgent",
                "generic_python_executor_tool",
                message=code_input.reasoning
                + "\n Pseudo Code: "
                + (code_input.pseudo_code or "N/A"),
                code=g_code,
                post_message=(
                    f"Output: {output[:max_output_length]}" + "..."
                    if len(output) > max_output_length
                    else ""
                ),
            )

            self._tool_call_counter_success["generic_python_executor_tool"] += 1
            return f"""
            Code:
            {g_code}

            Output:
            {output}
            """

        @function_tool
        async def attribute_lookup_tool(attribute_names_list: list[str]) -> str:
            """
            Look up the explanation for a given list of attribute names from the attribute_explanations dictionary.

            Args:
                attribute_names_list (list[str]): The names of the attributes to look up.

            Returns:
                str: The explanation for the attribute, or a message indicating it was not found.
            """

            try:
                self._tool_call_counter["attribute_lookup_tool"] += 1
                res = df_attribute_explanations[
                    df_attribute_explanations["Attribute"].isin(attribute_names_list)
                ]

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "attribute_lookup_tool",
                    f"Lookup attributes: {attribute_names_list}",
                )

                self._tool_call_counter_success["attribute_lookup_tool"] += 1
                return res
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error looking up attributes: {e}"
                )

        @function_tool
        async def attribute_pruning_tool(attribute_names_list: list[str]) -> str:
            """
            Prune the given list of attribute names from the attribute_explanations dictionary.

            Args:
                attribute_names_list (list[str]): The names of the attributes to prune.

            Returns:
                str: A message indicating the result of the pruning operation.
            """

            try:
                self._tool_call_counter["attribute_pruning_tool"] += 1

                self._num_features_pruned += (
                    df_attribute_explanations.loc[
                        df_attribute_explanations["Attribute"].isin(
                            attribute_names_list
                        ),
                        "Status",
                    ]
                    .ne("Pruned")
                    .sum()
                )

                df_attribute_explanations.loc[
                    df_attribute_explanations["Attribute"].isin(attribute_names_list),
                    "Status",
                ] = "Pruned"

                # remove rows where Attribute is in attribute_names_list
                # df_attribute_explanations.drop(
                #     df_attribute_explanations[
                #         df_attribute_explanations["Attribute"].isin(
                #             attribute_names_list
                #         )
                #     ].index,
                #     inplace=True,
                # )

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "attribute_pruning_tool",
                    f"Pruned attributes ({len(attribute_names_list)}): {attribute_names_list}",
                )

                self._tool_call_counter_success["attribute_pruning_tool"] += 1
                return "Attributes pruned successfully."
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error pruning attributes: {e}"
                )

        async def generate_report() -> str | None:

            agent = Agent(
                model_settings=ModelSettings(
                    temperature=test_agent_cfg.temperature,
                    reasoning=test_agent_cfg.reasoning,
                ),
                name="Tester Agent",
                instructions=f"""

                    # The Setup:
                    You are part of a team of agents working together to generate features with high predictive power.
                    The other agents in the team are:
                    - The Scientist Agent: {AgentRoles.SCIENTIST.value}
                    - The Extractor Agent: {AgentRoles.EXTRACTOR.value}

                    You all work in a loop where the Scientist Agent generates focus areas, the Extractor Agent extracts attributes, and you assess the features.

            
                    # Your Role and Tasks:
                    You are a Tester Agent tasked with assessing and evaluating the performance of aggregated features from the Extractor Agent.
                    You work autonomously to design and execute experiments that assess the usefulness of these features with respect to the following aspects:
                    - Predictive Power: Evaluate how well the features can predict the outlier patients that are not eligible for monitoring. This is the most important aspect.
                    - Feature Importance: Determine the importance of each feature in predicting the outlier patients that are not eligible for monitoring using appropriate techniques.
                    - Statistical Relationships: Analyze statistical inter-feature relationships. Assess correlations and interactions between features to identify redundancies or synergies.
                    - Impact Analysis: Investigate how different combinations of features affect the model's performance.
                    - Robustness Testing: Evaluate the robustness of the features under various conditions, such as noise addition or data perturbation.
                    
                    Your end goal is to provide a comprehensive report on the effectiveness of the features with respect to predicting the the outlier patients that are not eligible for monitoring.
                    Use the available tools to set up and run experiments, take notes, and retrieve information as needed.
                    Based on your findings, you may also prune features that do not contribute meaningfully to the prediction task by using the 'attribute_pruning_tool'.

                    # Your Workflow:
                    Follow the following steps:
                    0. Plan your approach to evaluate the features. Use the 'search_in_literature_tool' to get insights on relevant methodologies from the literature if needed.
                    1. Use the 'generic_python_executor_tool' to set up experiments using the provided feature datasets.
                    2. Use the 'take_note_tool' to document important observations and findings during the experiments.
                    3. Repeat steps 0, 1, and 2 as necessary to refine your experiments and gather insights.
                    4. Use the 'attribute_lookup_tool' to get explanations for specific attributes.
                    5. Use the 'attribute_pruning_tool' to prune a few features that are not useful based on your assessments. 
                    6. Perform step 1-5 iteratively until you are satisfied with your evaluation.
                    7. Use the 'get_notes_tool' to retrieve all your notes and observations.
                    8. Compile your findings into a comprehensive report that summarizes the performance of the features in predicting the target variable.
                    
                    # The Context and Global Goal:
                    {self.cfg_experiment.context_and_goal}

                    # Important Guidelines:
                    - Use the 'search_in_literature_tool' tool to look up relevant methodologies from the literature to inform your experimental design.
                    - Always think step-by-step and explain your reasoning.
                    - Your final report should be a concise summary of your findings, including key and noteworthy results from your experiments. Avoid unnecessary details.
                    - Use clear and precise language to communicate your results effectively.
                    - When using the 'generic_python_executor_tool' tool, ensure that your code is well-documented and easy to understand.
                    - DO NOT provide recommendations for feature engineering or data preprocessing. Focus solely on evaluating the features as they are provided.
                    - Prune features that do not contribute meaningfully to the prediction task using the 'attribute_pruning_tool'. The overall number of features should be kept manageable.
                    - When using XGBoost, always pass device="cuda:3" and tree_method="hist" in the model parameters to run on the appropriate GPU (much faster).
                    - All the needed data is already loaded and available for use in the 'df_attributes' pandas DataFrame. DO NOT SEARCH FOR FILES OR DATA PATHS.
                
                    """,
                model=OpenAIResponsesModel(
                    model=llm_cfg.model,
                    openai_client=self._client,
                ),
                tools=[
                    generic_python_executor_tool,
                    take_note_tool,
                    get_notes_tool,
                    attribute_lookup_tool,
                    search_in_literature_tool,
                    attribute_pruning_tool,
                ],
            )

            session = SQLiteSession(f"tester_agent_session_step_{step or 0}")

            for _ in range(20):
                try:

                    out = await Runner.run(
                        agent,
                        "",
                        max_turns=test_agent_cfg.max_iterations,
                        session=session,
                    )
                    if len(out.final_output) > 0:
                        return out.final_output

                except Exception as e:
                    ConsoleManager.console_error_print(f"Error generating report. {e}")
                    continue

        def compute_metrics_tabular() -> TestResultClassification:

            queue = multiprocessing.Manager().Queue()
            for idx in GPUS:
                queue.put(idx)

            # Calculate scale_pos_weight for class imbalance
            # num_pos = y_all_train["target"].sum()
            # num_neg = len(y_all_train) - num_pos
            # scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

            def objective(trial: optuna.Trial) -> float:
                # Define the search space
                params = {
                    # Fixed parameters
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "tree_method": "hist",
                    "random_state": 42,
                    # "scale_pos_weight": scale_pos_weight,
                    # Hyperparameters to tune
                    "n_estimators": trial.suggest_int("n_estimators", 100, 5000),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                }

                device = queue.get()

                model = xgb.XGBRegressor(**params, device=device)

                # model = xgb.XGBClassifier(**params, device=device)

                y_all_enc = y_all_train.copy()

                y_all_enc.drop(columns=["id"], inplace=True)
                y_all_enc = y_all_enc.astype(float)

                model.fit(X_all_train, y_all_enc)

                predictions_enc = model.predict(X_all_train)

                queue.put(device)

                return np.sqrt(mean_squared_error(y_all_enc, predictions_enc))

            try:

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=20, n_jobs=len(GPUS))
                    best_params = study.best_params
            except Exception as e:
                ConsoleManager.console_error_print(
                    f"Error during hyperparameter optimization: {e}"
                )
                best_params = {}

            df_best_params = pd.DataFrame([best_params])

            # device = queue.get()
            model = xgb.XGBRegressor(
                **best_params,
                objective="reg:squarederror",
                eval_metric="rmse",
                # scale_pos_weight=scale_pos_weight,
                random_state=42,
                device="cpu",
                tree_method="hist",
                # use_label_encoder=False,
            )
            # Ensure encoded labels are 1D integer arrays for class_weight utilities

            y_all_enc = y_all_train.copy()
            y_all_enc.drop(columns=["id"], inplace=True)
            y_all_enc = y_all_enc.astype(float)

            model.fit(X_all_train, y_all_enc)

            predictions_enc_test = model.predict(X_all_test)
            predictions_enc_train = model.predict(X_all_train)

            train_score = 0
            best_val = 0
            for val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                patient_ids_train_val = df_attributes_train[predictions_enc_train > val][
                    "id"
                ].values

                tmp_score = self._brainflux_filter_pipeline.get_score(
                    patient_ids_train_val.tolist()
                )["train"]
                if tmp_score > train_score:
                    train_score = tmp_score
                    best_val = val

            # Evaluate the chosen threshold on the held-out test partition
            patient_ids_test = df_attributes_test[predictions_enc_test > best_val][
                "id"
            ].values
            patient_ids_train = df_attributes_train[predictions_enc_train > best_val][
                "id"
            ].values
            score = self._brainflux_filter_pipeline.get_score(
                patient_ids_train.tolist()
            )["test"]
            patient_labels = self._brainflux_filter_pipeline.get_patient_classes(
                list_of_patients=patient_ids_test.tolist()
            )

            top_10_features = sorted(
                zip(df_attributes.columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[: min(10, len(df_attributes.columns))]
            top_10_features_dict = {name: score for name, score in top_10_features}

            out = df_best_params.copy()
            out["test_global_recall"] = score
            out["test_rmse"] = np.sqrt(
                mean_squared_error(y_all_test["target"], predictions_enc_test)
            )
            out["best_threshold"] = best_val

            ConsoleManager.print_dataframe_as_table(
                out, title="Best Hyperparameters", style="green"
            )

            # queue.put(device)

            return TestResultClassification(
                global_recall=score,
                loss=out["test_rmse"].iloc[0],
                best_threshold=out["best_threshold"].iloc[0],
                target_class_patients_removed=sum(
                    [
                        1
                        for label in patient_labels.values()
                        if label == self.target_class
                    ]
                ),
                non_target_class_patients_removed=sum(
                    [
                        1
                        for label in patient_labels.values()
                        if label != self.target_class
                    ]
                ),
                top_impactful_attributes=top_10_features_dict,
            )

        report = await generate_report()

        df_attributes = self.prune_attributes_in_df(
            df_attributes, df_attribute_explanations
        )

        results = compute_metrics_tabular()

        if report:
            results.report = report

        @wandb_logging_wrapper
        async def log_results_to_wandb():
            import wandb
            import tempfile
            import os

            # Log metrics
            wandb.log({"test_results": results.to_dict(only_metrics=True)}, step=step)

            wandb.log(
                {"Num Features Pruned": self._num_features_pruned},
                step=step,
            )

            # Log assessment report
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md")
                tmp_path = tmp.name
                tmp.write(results.report or "")
                tmp.close()

                artifact = wandb.Artifact(
                    name=f"Test_assessment_report_Trail_{step}", type="report"
                )
                artifact.add_file(tmp_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                ConsoleManager.console_error_print(
                    f"Failed to create/log temp artifact: {e}"
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # Log Feature Count
            wandb.log(
                {"feature_count": df_attributes.shape[1] - 1},  # exclude target column
                step=step,
            )

            # Log Tool Usage
            wandb.log(
                {
                    f"Tester_tool_usage/{tool_name}": count
                    for tool_name, count in self._tool_call_counter.items()
                },
                step=step,
            )

            wandb.log(
                {
                    f"Tester_tool_usage_success/{tool_name}": count
                    for tool_name, count in self._tool_call_counter_success.items()
                },
                step=step,
            )

        await log_results_to_wandb()

        return results, df_attribute_explanations
