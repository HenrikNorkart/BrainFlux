import asyncio
import contextlib
from dataclasses import dataclass
import multiprocessing
from textwrap import dedent
import io
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, Field
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, KFold
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

from rogueone.utils import llm_cfg, rouge_one_cfg, test_agent_cfg
from rogueone.utils.console import ConsoleManager
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.wandb_utils import wandb_logging_wrapper
from rogueone.llm.prompts.agent_roles import AgentRoles
from rogueone.llm.agents.knowledge import KnowledgeAgent
from rogueone.llm.agents.web_search import WebSearchAgent

# set_tracing_disabled(True)

GPUS = [
    "cuda:3",
    "cuda:4",
    "cuda:5",
    "cuda:6",
    "cuda:7",
]


@dataclass
class TestResultRegression:
    mses: list[float]
    maes: list[float]
    r2s: list[float]
    max_val: float
    min_val: float
    mean_val: float
    report: str = ""

    @property
    def value_range(self) -> float:
        return abs(self.max_val - self.min_val)

    @property
    def n_rmse(self) -> float:
        return float(np.mean(self.n_rmses))

    @property
    def n_rmse_std(self) -> float:
        return float(np.std(self.n_rmses))

    @property
    def n_rmses(self) -> list[float]:
        return [rmse / self.mean_val for rmse in self.rmses]

    @property
    def rmses(self) -> list[float]:
        return [np.sqrt(mse) for mse in self.mses]

    @property
    def rmse(self) -> float:
        return float(np.mean(self.rmses))

    @property
    def rmses_std(self) -> float:
        return float(np.std(self.rmses))

    @property
    def mse(self) -> float:
        return float(np.mean(self.mses))

    @property
    def mse_std(self) -> float:
        return float(np.std(self.mses))

    @property
    def mae(self) -> float:
        return float(np.mean(self.maes))

    @property
    def mae_std(self) -> float:
        return float(np.std(self.maes))

    @property
    def r2(self) -> float:
        return float(np.mean(self.r2s))

    @property
    def r2_std(self) -> float:
        return float(np.std(self.r2s))

    @property
    def as_dict(self) -> dict:
        return {
            "n_RMSE": self.n_rmse,
            "RMSE": self.rmse,
            "MSE": self.mse,
            "MAE": self.mae,
            "R2": self.r2,
            "report": self.report,
        }

    def to_dict(self, only_metrics: bool = True) -> dict:
        if only_metrics:
            return {
                "n_RMSE": self.n_rmse,
                "MSE": self.mse,
                "MAE": self.mae,
                "R2": self.r2,
            }
        else:
            return self.as_dict


@dataclass
class TestResultClassification:
    accuracys: list[float]
    precisions: list[float]
    recalls: list[float]
    f1s: list[float]
    aurocs: list[float]
    report: str = ""

    # feature_importance: dict[str, float]

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.accuracys))

    @property
    def accuracy_std(self) -> float:
        return float(np.std(self.accuracys))

    @property
    def precision(self) -> float:
        return float(np.mean(self.precisions))

    @property
    def precision_std(self) -> float:
        return float(np.std(self.precisions))

    @property
    def recall(self) -> float:
        return float(np.mean(self.recalls))

    @property
    def recall_std(self) -> float:
        return float(np.std(self.recalls))

    @property
    def f1(self) -> float:
        return float(np.mean(self.f1s))

    @property
    def f1_std(self) -> float:
        return float(np.std(self.f1s))

    @property
    def auroc(self) -> float:
        return float(np.mean(self.aurocs))

    @property
    def auroc_std(self) -> float:
        return float(np.std(self.aurocs))

    @property
    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "AUROC": self.auroc,
            "report": self.report,
        }

    @property
    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.as_dict])

    def to_dict(self, only_metrics: bool = True) -> dict:
        if only_metrics:
            return {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "AUROC": self.auroc,
            }
        else:
            return self.as_dict

    def __repr__(self):
        return dedent(
            f"""
            Accuracy: {self.accuracy}
            Precision: {self.precision}
            Recall: {self.recall}
            F1 Score: {self.f1}
            AUROC: {self.auroc}
            Report: \n{self.report}
        """
        )


class CodeInput(BaseModel):
    code: str = Field(..., description="The python code to execute.")
    reasoning: str = Field(
        ...,
        description="The reasoning behind the code execution. Explain why this code is being executed and what it aims to achieve.",
    )


class TesterAgent:

    def __init__(self, cfg: ExperimentConfig, *, include_report: bool = True):
        self.cfg_experiment = cfg
        self.include_report = include_report
        self.knowledge_agent = WebSearchAgent(
            cfg=self.cfg_experiment  # , collection_name="tester_knowledge_collection"
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
    ) -> tuple[TestResultClassification | TestResultRegression, pd.DataFrame]:

        match self.cfg_experiment.modality:
            case "tabular":

                df_attributes_folds = [
                    d.drop("id", axis=1) for d in df_attributes_folds
                ]

                x_folds = [
                    d.drop("target", axis=1).to_numpy() for d in df_attributes_folds
                ]
                y_folds = [d["target"].to_numpy() for d in df_attributes_folds]

                df_attributes = pd.concat(
                    df_attributes_folds, ignore_index=True
                ).reset_index(drop=True)

            case "time_series":
                df_train, df_test = df_attributes_folds
                df_train = df_train.drop("id", axis=1)
                df_test = df_test.drop("id", axis=1)

                df_attributes = pd.concat(
                    [df_train.drop(["is_test", "is_train"], axis=1)], axis=0
                ).reset_index(drop=True)

                X_train = df_train.drop(
                    ["target", "is_test", "is_train"], axis=1
                ).to_numpy()
                y_train = df_train["target"].to_numpy()

                X_test = df_test.drop(
                    ["target", "is_test", "is_train"], axis=1
                ).to_numpy()
                y_test = df_test["target"].to_numpy()

            case _:
                raise NotImplementedError(
                    f"Modality {self.cfg_experiment.modality} not supported yet."
                )

        df_attributes = self.prune_attributes_in_df(
            df_attributes, df_attribute_explanations
        )

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._num_features_pruned = 0

        _note_book = ""

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
            The environment has access to a attribute data (df_attributes) as pandas DataFrames.
            All code is non persistent and stateless between calls.

            The environment have the following libraries imported and available for use:
            - pandas as pd
            - numpy as np
            - sklearn
            - shap
            - statsmodels as sm
            - scipy as spy
            - xgboost as xgb

            The dataframe has the following structure:
            - Each row represents an instance.
            - Each column represents a feature/attribute, except for the 'target' column which is the target variable to predict.

            The output of the code should be assigned to a variable named 'output'.

            Returns:
                str: The output of the executed code or an error message if execution fails.
            """

            try:
                self._tool_call_counter["generic_python_executor_tool"] += 1

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    local_vars = {}
                    exec(
                        code_input.code,
                        {
                            "df_attributes": df_attributes,
                            "pd": pd,
                            "np": np,
                            "sklearn": sklearn,
                            "shap": shap,
                            "sm": sm,
                            "spy": spy,
                            "xgb": xgb,
                        },
                        local_vars,
                    )

                    output = local_vars.get("output", "No output variable defined.")

                output = f"{output}"

                max_output_length = min(len(output), 500)

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "generic_python_executor_tool",
                    message=code_input.reasoning,
                    code=code_input.code,
                    post_message=(
                        f"Output: {output[:max_output_length]}" + "..."
                        if len(output) > max_output_length
                        else ""
                    ),
                )

                self._tool_call_counter_success["generic_python_executor_tool"] += 1
                return output

            except Exception as e:

                return ConsoleManager.console_error_print(
                    f"Error executing the code. Please check your syntax and try again: {e}",
                    code=code_input.code,
                )

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

        async def generate_report():

            client = AsyncOpenAI(
                base_url=llm_cfg.endpoint,
                api_key=llm_cfg.api_key,
                organization="brainflux-inc",
                project="brainflux_project",
                webhook_secret=None,
            )

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
                    - Predictive Power: Evaluate how well the features can predict the target variable 'target'.
                    - Feature Importance: Determine the importance of each feature in predicting the target variable using appropriate techniques.
                    - Statistical Relationships: Analyze statistical inter-feature relationships. Assess correlations and interactions between features to identify redundancies or synergies.
                    - Impact Analysis: Investigate how different combinations of features affect the model's performance.
                    - Robustness Testing: Evaluate the robustness of the features under various conditions, such as noise addition or data perturbation.
                    
                    Your end goal is to provide a comprehensive report on the effectiveness of the features with respect to predicting the 'target' variable.
                    Use the available tools to set up and run experiments, take notes, and retrieve information as needed.
                    Based on your findings, you may also prune features that do not contribute meaningfully to the prediction task by using the 'attribute_pruning_tool'.

                    # Your Workflow:
                    Follow the following steps:
                    0. Plan your approach to evaluate the features. Use the 'search_in_literature_tool' to get insights on relevant methodologies from the literature if needed.
                    1. Use the 'generic_python_executor_tool' to set up experiments using the provided feature datasets.
                    2. Use the 'take_note_tool' to document important observations and findings during the experiments.
                    3. Repeat steps 0, 1, and 2 as necessary to refine your experiments and gather insights.
                    4. Use the 'attribute_lookup_tool' to get explanations for specific attributes.
                    5. Use the 'attribute_pruning_tool' to prune features that are not useful based on your assessments. 
                    6. Compile your findings into a comprehensive report that summarizes the performance of the features in predicting the target variable.
                    
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
                    - When using XGBoost, always pass device="cpu" and tree_method="hist" in the model parameters to run on CPU.
                
                    """,
                model=OpenAIResponsesModel(
                    model=llm_cfg.model,
                    openai_client=client,
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

        def get_regression_metrics() -> TestResultRegression:
            queue = multiprocessing.Manager().Queue()
            for idx in GPUS:
                queue.put(idx)

            def objective(trial: optuna.Trial) -> float:
                # Define the search space
                params = {
                    # Fixed parameters
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "tree_method": "hist",
                    "random_state": 42,
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

                X_all = np.vstack([x_folds[j] for j in range(len(x_folds))])
                y_all = np.hstack([y_folds[j] for j in range(len(x_folds))])

                kfold = KFold(
                    n_splits=rouge_one_cfg.k_folds, shuffle=True, random_state=42
                )
                score = cross_val_score(
                    model,
                    X_all,
                    y_all,
                    cv=kfold,
                    scoring="neg_root_mean_squared_error",
                ).mean()

                queue.put(device)

                return score

            with contextlib.redirect_stdout(io.StringIO()) as f:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, n_jobs=len(GPUS))

            best_params = study.best_params

            df_best_params = pd.DataFrame([best_params])
            ConsoleManager.print_dataframe_as_table(
                df_best_params, title="Best Hyperparameters", style="green"
            )

            mses = []
            maes = []
            r2s = []

            for i in range(len(x_folds)):
                X_test = x_folds[i]
                y_test = y_folds[i]

                X_train = np.vstack([x_folds[j] for j in range(len(x_folds)) if j != i])
                y_train = np.hstack([y_folds[j] for j in range(len(x_folds)) if j != i])

                model = xgb.XGBRegressor(
                    **best_params,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=42,
                    device="cuda:5",
                    tree_method="hist",
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mses.append(mean_squared_error(y_test, y_pred))
                maes.append(mean_absolute_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))

            return TestResultRegression(
                mses=mses,
                maes=maes,
                r2s=r2s,
                max_val=df_attributes["target"].max(),
                min_val=df_attributes["target"].min(),
                mean_val=df_attributes["target"].mean(),
            )

        def get_classification_metrics_tabular() -> TestResultClassification:

            queue = multiprocessing.Manager().Queue()
            for idx in GPUS:
                queue.put(idx)

            def objective(trial: optuna.Trial) -> float:
                # Define the search space
                params = {
                    # Fixed parameters
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "random_state": 42,
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

                model = xgb.XGBClassifier(**params, device=device)

                X_all = np.vstack([x_folds[j] for j in range(len(x_folds))])
                y_all = np.hstack([y_folds[j] for j in range(len(x_folds))])

                le = LabelEncoder()
                le.fit(y_all)
                y_all_enc = np.asarray(le.transform(y_all)).ravel().astype(int)

                # Use cross-validation to get a robust accuracy score
                kfold = KFold(
                    n_splits=rouge_one_cfg.k_folds, shuffle=True, random_state=42
                )
                score = cross_val_score(
                    model, X_all, y_all_enc, cv=kfold, scoring="accuracy"
                ).mean()

                queue.put(device)

                return score

            with contextlib.redirect_stdout(io.StringIO()) as f:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, n_jobs=len(GPUS))

            best_params = study.best_params

            df_best_params = pd.DataFrame([best_params])
            ConsoleManager.print_dataframe_as_table(
                df_best_params, title="Best Hyperparameters", style="green"
            )

            accs = []
            pres = []
            recs = []
            f1s = []
            aurocs = []

            for i in range(len(x_folds)):
                X_test = x_folds[i]
                y_test = y_folds[i]

                X_train = np.vstack([x_folds[j] for j in range(len(x_folds)) if j != i])
                y_train = np.hstack([y_folds[j] for j in range(len(x_folds)) if j != i])

                model = xgb.XGBClassifier(
                    **best_params,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    device="cuda:5",
                    tree_method="hist",
                    # use_label_encoder=False,
                )

                # Encode string labels to integers for XGBoost
                le = LabelEncoder()
                y_all = np.concatenate([y_train, y_test])
                le.fit(y_all)
                y_train_enc = np.asarray(le.transform(y_train)).ravel().astype(int)
                y_test_enc = np.asarray(le.transform(y_test)).ravel().astype(int)

                # Ensure encoded labels are 1D integer arrays for class_weight utilities
                y_train_enc = y_train_enc.astype(int)
                y_test_enc = y_test_enc.astype(int)

                sample_weights_train = (
                    class_weight.compute_sample_weight("balanced", y=y_train_enc)
                    if rouge_one_cfg.do_balancing
                    else None
                )

                model.fit(X_train, y_train_enc, sample_weight=sample_weights_train)

                predictions_enc = model.predict(X_test)

                sample_weights_test = (
                    class_weight.compute_sample_weight("balanced", y=y_test_enc)
                    if rouge_one_cfg.do_balancing
                    else None
                )

                # Binarize y_test for multi-class AUROC using encoded classes
                classes_enc = np.unique(np.concatenate([y_train_enc, y_test_enc]))
                y_test_binarized = label_binarize(y_test_enc, classes=classes_enc)

                proba = model.predict_proba(X_test)

                proba = np.asarray(proba)

                accs.append(
                    accuracy_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                    ),
                )
                pres.append(
                    precision_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                        average="weighted",
                    ),
                )
                recs.append(
                    recall_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                        average="weighted",
                    ),
                )
                f1s.append(
                    f1_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                        average="weighted",
                    ),
                )

                try:
                    # If binary (2 columns), extract probability for the positive class as a 1D array.
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        try:
                            pos_idx = list(model.classes_).index(1)
                        except Exception:
                            pos_idx = 1
                        proba = proba[:, pos_idx]
                    else:
                        proba = proba.ravel()

                    if len(classes_enc) == 2:
                        aurocs.append(
                            roc_auc_score(
                                y_test_enc,
                                proba,
                                average="weighted",
                            ),
                        )
                    else:
                        aurocs.append(
                            roc_auc_score(
                                y_test_binarized,
                                proba,
                                multi_class="ovr",
                                average="weighted",
                            ),
                        )
                except Exception as e:
                    ConsoleManager.console_error_print(
                        f"Error calculating AUROC for fold {i+1}: {e}"
                    )
                    aurocs.append(0.0)

            results = pd.DataFrame(
                {
                    "fold": [
                        f"Fold {i}/{len(accs)}" for i in np.arange(1, len(accs) + 1)
                    ],
                    "accuracy": accs,
                    "precision": pres,
                    "recall": recs,
                    "f1": f1s,
                    "AUROC": aurocs,
                }
            )

            avg_row = {
                "fold": "Average",
                "accuracy": float(np.mean(accs)),
                "precision": float(np.mean(pres)),
                "recall": float(np.mean(recs)),
                "f1": float(np.mean(f1s)),
                "AUROC": float(np.mean(aurocs)),
            }
            std_row = {
                "fold": "Std Dev",
                "accuracy": float(np.std(accs)),
                "precision": float(np.std(pres)),
                "recall": float(np.std(recs)),
                "f1": float(np.std(f1s)),
                "AUROC": float(np.std(aurocs)),
            }

            results = pd.concat(
                [results, pd.DataFrame([avg_row, std_row])], ignore_index=True
            )
            ConsoleManager.print_dataframe_as_table(
                results,
                title="Classification Metrics per Fold",
                style="green",
            )

            return TestResultClassification(
                accuracys=accs,
                precisions=pres,
                recalls=recs,
                f1s=f1s,
                aurocs=aurocs,
            )

        def get_classification_metrics_time_series() -> TestResultClassification:

            assert (
                X_train is not None
                and X_test is not None
                and y_train is not None
                and y_test is not None
            )

            queue = multiprocessing.Manager().Queue()
            for idx in GPUS:
                queue.put(idx)

            y_all = np.concatenate([y_train, y_test])

            le = LabelEncoder()
            le.fit(y_all)

            y_test_enc = np.asarray(le.transform(y_test)).ravel().astype(int)
            y_train_enc = np.asarray(le.transform(y_train)).ravel().astype(int)

            def objective(trial: optuna.Trial) -> float:
                # Define the search space
                params = {
                    # Fixed parameters
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "random_state": 42,
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

                # params = {
                #     # Hyperparameters to tune
                #     "n_estimators": trial.suggest_int("n_estimators", 100, 5000),
                #     "max_depth": trial.suggest_int("max_depth", 3, 15),
                #     "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                #     "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                #     "max_features": trial.suggest_categorical(
                #         "max_features", ["sqrt", "log2"]
                #     ),
                # }

                device = queue.get()

                # model = RandomForestClassifier(**params)

                model = xgb.XGBClassifier(**params, device=device)

                model.fit(X_train, y_train_enc)
                predictions_enc = model.predict(X_test)

                queue.put(device)

                return accuracy_score(y_test_enc, predictions_enc)

            with contextlib.redirect_stdout(io.StringIO()) as f:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, n_jobs=len(GPUS))

            best_params = study.best_params

            # model = RandomForestClassifier(**best_params)

            model = xgb.XGBClassifier(
                **best_params,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                device="cuda:5",
                tree_method="hist",
                # use_label_encoder=False,
            )

            model.fit(X_train, y_train_enc)
            predictions_enc = model.predict(X_test)
            try:
                proba = model.predict_proba(X_test)
                # Handle binary and multiclass probability shapes robustly
                if isinstance(proba, np.ndarray) and proba.ndim == 2:
                    if proba.shape[1] == 2:
                        # Binary case: select probability for the positive class (prefer label 1 if present)
                        try:
                            pos_idx = list(model.classes_).index(1)
                        except Exception:
                            pos_idx = 1
                        aurec = roc_auc_score(y_test_enc, proba[:, pos_idx])
                    else:
                        # Multiclass case: binarize true labels and pass full proba matrix
                        classes_enc = np.unique(
                            np.concatenate([y_train_enc, y_test_enc])
                        )
                        y_test_binarized = label_binarize(
                            y_test_enc, classes=classes_enc
                        )
                        aurec = roc_auc_score(
                            y_test_binarized,
                            proba,
                            multi_class="ovr",
                            average=None,
                        )
                else:
                    # Fallback if predict_proba returns unexpected shape/type
                    aurec = 0.0
            except Exception as e:
                ConsoleManager.console_error_print(f"Error calculating AUROC: {e}")
                aurec = 0.0
            return TestResultClassification(
                accuracys=[accuracy_score(y_test_enc, predictions_enc)],
                precisions=[precision_score(y_test_enc, predictions_enc, average=None)],
                recalls=[recall_score(y_test_enc, predictions_enc, average=None)],
                f1s=[
                    f1_score(y_test_enc, predictions_enc, average=None)
                ],  # , average="weighted")],
                aurocs=[aurec],
            )

        report = await generate_report()

        df_attributes = self.prune_attributes_in_df(
            df_attributes, df_attribute_explanations
        )

        match self.cfg_experiment.task_type:
            case "regression":
                results = get_regression_metrics()

            case "classification":
                if self.cfg_experiment.modality == "tabular":
                    results = get_classification_metrics_tabular()
                elif self.cfg_experiment.modality == "time_series":
                    results = get_classification_metrics_time_series()
                else:
                    raise NotImplementedError(
                        f"Modality {self.cfg_experiment.modality} not supported yet."
                    )
            case _:
                raise ValueError(f"Unknown task type: {self.cfg_experiment.task_type}")

        if self.include_report:
            results.report = report
        else:
            results.report = ""

        @wandb_logging_wrapper
        def log_results_to_wandb():
            import wandb
            import matplotlib.pyplot as plt
            import tempfile
            import os

            # Log metrics
            wandb.log({"test_results": results.to_dict(only_metrics=True)}, step=step)

            match self.cfg_experiment.task_type:
                case "regression":
                    for i in range(len(results.maes)):
                        wandb.log(
                            {
                                f"fold_metrics_{i}": {
                                    "MSE": results.mses[i],
                                    "MAE": results.maes[i],
                                    "R2": results.r2s[i],
                                    "RMSE": results.rmses[i],
                                    "NRMSE": results.n_rmses[i],
                                }
                            },
                            step=step,
                        )

                case "classification":
                    for i in range(len(results.accuracys)):
                        wandb.log(
                            {
                                f"fold_metrics_{i}": {
                                    "accuracy": results.accuracys[i],
                                    "precision": results.precisions[i],
                                    "recall": results.recalls[i],
                                    "f1": results.f1s[i],
                                    "AUROC": results.aurocs[i],
                                }
                            },
                            step=step,
                        )

                case _:
                    raise ValueError(
                        f"Unknown task type: {self.cfg_experiment.task_type}"
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

            # # t-SNE plot
            # tsne = TSNE(n_components=2, random_state=42)
            # X_2d = tsne.fit_transform(np.vstack([X_train, X_test]))
            # y_combined = np.hstack([y_train, y_test], dtype=int)

            # unique_labels = np.unique(y_combined)
            # colors = plt.cm.Set1(np.linspace(0, 1, 10))

            # plt.figure(figsize=(6, 6))
            # for lbl, col in zip(unique_labels, colors):
            #     mask = y_combined == lbl
            #     plt.scatter(
            #         X_2d[mask, 0],
            #         X_2d[mask, 1],
            #         color=col,
            #         s=20,
            #         alpha=0.8,
            #     )

            # plt.title("t-SNE Scatter Plot")
            # plt.xlabel("TSNE 1")
            # plt.ylabel("TSNE 2")

            # wandb.log({"TSNE Scatter": plt}, step=step)
            # plt.close()

        log_results_to_wandb()

        return results, df_attribute_explanations


if __name__ == "__main__":

    from rogueone.utils.wandb_utils import init_wandb_run
    from rogueone.utils.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(
        "/workspaces/BrainFlux/tasks/suppression_rato/config.yml"
    )

    init_wandb_run(cfg)

    console_manager = ConsoleManager
    console_manager.console_rule("Tester Agent Test", color="red")

    df_patients = pd.read_csv("/workspaces/BrainFlux/tmp/df_patients.csv")

    from rogueone.utils.train_test_splitter import TrainTestSplitter
    from pathlib import Path

    splitter = TrainTestSplitter(
        labels_file_path=cfg.label_file_path,
        test_size=0.5,
        random_state=42,
    )

    patient_attributes_train, patient_attributes_test = splitter.test_train_split(
        df_patients.copy()
    )

    console_manager.console_print(
        f"Training set size: {len(patient_attributes_train)}, Test set size: {len(patient_attributes_test)}"
    )

    tester_agent = TesterAgent(cfg)

    json_path = Path("/workspaces/BrainFlux/tmp/attribute_descriptions.json")

    try:
        try:
            attribute_explanations = pd.read_json(json_path, lines=True)
        except ValueError:
            attribute_explanations = pd.read_json(json_path)

    except Exception as e:
        console_manager.console_error_print(f"Failed to load JSON {json_path}: {e}")

    for i in range(3):
        test_result = tester_agent.test_hypotheses(
            patient_attributes_train,
            patient_attributes_test,
            attribute_explanations,
            step=i,
        )

        console_manager.console_rule(f"Test Result Iteration {i+1}", color="blue")

        result_dict = test_result.as_dict.copy()
        result_dict.pop("report", None)

        console_manager.print_dict_as_table(result_dict)

        console_manager.console_print(test_result.report)
