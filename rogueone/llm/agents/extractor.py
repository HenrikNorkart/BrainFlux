# %%

import io
import contextlib
import asyncio
from logging import getLogger
from collections import defaultdict
from textwrap import dedent

import numpy as np
from pydantic import BaseModel, Field
import pandas as pd
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIResponsesModel,
    ModelSettings,
    function_tool,
)
import scipy.signal as signal
import tsfresh
from agents.memory.sqlite_session import SQLiteSession

from rogueone.dataclasses.attributes import AttributeExplanation
from rogueone.utils.console import ConsoleManager
from rogueone.utils import llm_cfg, extractor_agent_cfg
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.wandb_utils import wandb_logging_wrapper
from rogueone.llm.prompts.agent_roles import AgentRoles
from rogueone.llm.agents.knowledge import KnowledgeAgent
from rogueone.llm.agents.web_search import WebSearchAgent


# set_tracing_disabled(True)


logger = getLogger(__name__)


class PandasCommandArgs(BaseModel):
    command: str = Field(
        ...,
        description="The pandas command to used to analyze the data. This command should return a pandas Series or DataFrame with the same length as df_features when executed. The command should be a valid pandas command starting with 'df', 'df_features', or 'df_time_series' as the dataframe variable.",
    )
    explanation: str = Field(
        ...,
        description="""
            A brief explanation of what the command does and a justification for why it is useful for extracting patient attributes.
            Answer how this command helps in identifying relevant attributes based on the focus area provided by the Scientist Agent.
        """,
    )


class AttributeExplainingArgs(BaseModel):
    name: str = Field(..., description="The name of the attribute to explain.")
    command: str = Field(
        ...,
        description="The pandas command to used to calculate the attribute. This command should return a pandas Series or DataFrame with the same length as df_features when executed. Make sure to include a groupby operation on 'id' in the command.",
    )
    argumentation: str = Field(
        ...,
        description="The reasoning behind why the attribute should be added. Use this to explain the clinical relevance of the attribute.",
    )


class ExtractorAgent:

    def __init__(self, cfg: ExperimentConfig):

        self.cfg_experiment = cfg
        self.knowledge_agent = WebSearchAgent(
            cfg=self.cfg_experiment  # , collection_name="feature_extraction_literature"
        )

    async def generate_patient_attributes(
        self,
        _df: pd.DataFrame,
        _df_patients: pd.DataFrame,
        focus: str,
        *,
        step: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate patient attributes based on the provided DataFrames and focus.

        Args:
            df (pd.DataFrame): The main DataFrame containing patient data.
            df_features (pd.DataFrame): The DataFrame containing patient identifiers.
            focus (str): The specific focus area for attribute extraction.

        Returns:
            tuple[pd.DataFrame, dict]: A tuple containing the extracted patient attributes DataFrame and a dictionary of attribute explanations.
        """

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)

        try:
            # Convert the 'datetime' column to pandas datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(_df["time"]):
                _df["datetime"] = pd.to_datetime(_df["time"], errors="coerce")
        except:
            pass

        df = _df.copy()
        df_features = _df_patients.copy()
        try:
            df_time_series = df.set_index("datetime")
        except:
            df_time_series = "No time series data available."

        attribute_descriptions: list[AttributeExplanation] = []

        know_attributes = set(df_features.columns)

        @function_tool
        async def search_in_literature_tool(query: str) -> str:
            """
            Search up and get a summary from relevant literature to help expand your knowledge.
            Use this tool to learn more about feature extraction techniques so that you can create better features.
            The tool does an online search and provides a summary of the most relevant information found.
            It can also be used to look up documentation for external libraries or frameworks like, pandas, tsfresh and scipy.signal.

            Args:
                query (str): The search query.
            Returns:
                str: A summary of the explanation.
            """
            try:
                self._tool_call_counter["search_in_literature_tool"] += 1
                explanation = await self.knowledge_agent.explain_query(query)

                # # Support both synchronous and asynchronous implementations of explain_query.
                # result = self.knowledge_agent.explain_query(query)
                # if asyncio.iscoroutine(result):
                #     explanation = await result
                # else:
                #     explanation = result

                # Normalize and check for empty responses
                explanation = explanation if explanation is not None else ""
                if str(explanation).strip() == "":
                    return ConsoleManager.console_error_print(
                        f"No explanation found for query: {query}"
                    )

                ConsoleManager.console_agent_logging(
                    "ScientistAgent", "search_in_literature_tool", query
                )
                self._tool_call_counter_success["search_in_literature_tool"] += 1
                return explanation
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error during literature search: {e}"
                )

        @function_tool
        async def generic_pandas_tool(args: PandasCommandArgs) -> str:
            # The command should be a valid pandas command using 'df' as the dataframe variable for standard operations.
            """
            Execute a generic pandas command and return the result.
            The command should be a valid pandas command using 'df', 'df_features', or 'df_time_series' as the dataframe variable.
            The pandas package is already imported as 'pd' and numpy as 'np'.
            All commands must:
                - contain 'df', 'df_features', or 'df_time_series'
                - store the output in a variable named 'result'

            # For Time series analysis:
                Use the 'df_time_series' dataframe which has the 'datetime' column set as the index.
                The time series libraries scipy.signal and tsfresh are also imported as 'signal' and 'tsfresh' respectively.

            # IMPORTANT:
                EDITS TO DATAFRAMES ARE NOT PERMANENT AND WILL NOT BE SAVED.

            # Tool Functionality:
            The tool works as follows:
                ```
                exec(
                        f"{args.command}",
                        {
                            "pd": pd,
                            "df": df,
                            "np": np,
                            "signal": signal,
                            "tsfresh": tsfresh,
                            "df_features": df_features,
                            "df_time_series": df_time_series,
                        },
                        locals_dict,
                    )
                return locals_dict.get("result", None)
                ```

            # Example of valid commands:
                - result = df.head()
                - result = df.describe()
                - result = df['column_name'].value_counts()
                - result = df_features['id'].nunique()
                - result = df_time_series.resample('D').mean()
                - tmp = df.groupby('id')['lab_value'].mean(); result = tmp.fillna(0)

            # Example of invalid commands:
                - df = df.dropna()  # Reassigning df is not allowed
                - df[df['column'] > 0]  # Not storing result in 'result' variable is not allowed
                - df_features = df_features.merge(other_df)  # Reassigning df_features is not allowed
                - df_time_series['new_col'] = df_time_series['col1'] + df_time_series['col2']  # Creating new columns is not allowed

            # Args:
                command (PandasCommandArgs): The pandas command to execute. It should be a valid pandas command using 'df', 'df_features', or 'df_time_series' as the dataframe variable.
            # Returns:
                str: The result of the pandas command as a string.

            """
            try:
                self._tool_call_counter["generic_pandas_tool"] += 1

                if (
                    args.command.strip().startswith("df =")
                    or args.command.strip().startswith("df_features =")
                    or args.command.strip().startswith("df_time_series =")
                    or not "df" in args.command.strip()
                ):
                    return ConsoleManager.console_print(
                        f"[red]ILLEGAL COMMAND: Command must a valid pandas command containing 'df', 'df_features', or 'df_time_series'.[/red]\nCommand: {args.command}"
                    )

                ConsoleManager.console_agent_logging(
                    self.__class__.__name__,
                    "generic_pandas_tool",
                    message=args.explanation,
                    code=args.command,
                )

                # Use eval to execute the command in the context of the dataframe

                locals_dict = {}

                with contextlib.redirect_stdout(io.StringIO()) as g:
                    exec(
                        f"{args.command}",
                        {
                            "pd": pd,
                            "df": df,
                            "np": np,
                            "signal": signal,
                            "tsfresh": tsfresh,
                            "df_features": df_features,
                            "df_time_series": df_time_series,
                        },
                        locals_dict,
                    )

                result = locals_dict.get("result", None)

                if result is None and g.getvalue().strip() == "":
                    return ConsoleManager.console_error_print(
                        f"Error: Command did not set 'result' variable or 'result' was set to None! Command: {args.command}"
                    )

                # If the result is a DataFrame or Series, convert it to string for better readability
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    result = result.to_string()

                out = dedent(
                    f"""
                    Result:
                    {result}    

                    Terminal Output:
                    {g.getvalue() if g.getvalue().strip() else "No output generated."}                    
                    """
                ).strip()

                self._tool_call_counter_success["generic_pandas_tool"] += 1
                return out

            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error executing command: {e}", code=args.command
                )

        @function_tool
        async def append_new_attribute(args: AttributeExplainingArgs) -> str:
            """
            Explain a patient attribute and its calculation method.
            This tool should be used to document new attributes added that should be added to the 'df_features' dataframe.
            The tool adds attributes to the 'df_features' dataframe by running the following code from the tool arguments:
                ```
                locals_dict = {}
                exec(
                    f"{args.command}",
                    {
                        "pd": pd,
                        "df": df,
                        "np": np,
                        "signal": signal,
                        "tsfresh": tsfresh,
                        "df_features": df_features,
                        "df_time_series": df_time_series,
                    },
                    locals_dict,
                )
                df_features[args.name] = locals_dict.get("result")
                ```
            # For Time series analysis:
                Use the 'df_time_series' dataframe which has the 'datetime' column set as the index.
                The time series libraries scipy.signal and tsfresh are also imported as 'signal' and 'tsfresh' respectively.
                These can be used to create more complex time series based attributes like trends, seasonality, peaks, frequency components, etc.

            In other words: args.commands should return a pandas dataframe or series with the same length as df_features when executed.
            This is done by including a groupby operation on 'id' in the command.
            The command should:
            - Be a valid pandas command.
            - Be using 'df', 'df_features', or 'df_time_series' as the dataframe variable.
            - Not reassign 'df', 'df_features', or 'df_time_series' (e.g., df = ...).
            - Store the output in a variable named 'result'. The variable 'result' is then added to 'df_features' as a new column with the name specified in 'args.name'.
            - The output stored in 'result' must be numeric (integer or float) and must not contain any NaN or Inf values and must be the same length as df_features.

            # IMPORTANT:
            - Not create new dataframes. Only read, do not write.
            - The command must include a groupby operation on 'id' to ensure alignment with df_features.
            - The tool works! Do not test it with dummy commands.

            The pandas package is already imported as 'pd'.

            # Example of valid commands:
                - result = df.groupby('id')['lab_value'].mean()
                - result = df_time_series.groupby('id')['heart_rate'].max()
                - result = df_features['id'].map(df.groupby('id')['age'].first())
            # Example of a valid multi-statement command:
                - tmp = df.groupby('id')['lab_value'].mean(); result = tmp.fillna(0)
            # Example of invalid commands:
                - df = df.dropna()  # Reassigning df is not allowed
                - df_features = df_features.merge(other_df)  # Reassigning df_features is not allowed
                - df_time_series['new_col'] = df_time_series['col1'] + df_time_series['col2']  # Creating new columns is not allowed

            # Args:
                args (AttributeExplainingArgs): The arguments containing attribute information.

            # Returns:
                str: A confirmation message.
            """
            try:
                self._tool_call_counter["append_new_attribute"] += 1

                if args.name in know_attributes:
                    ConsoleManager.console_print(
                        f"[red]Error: Attribute {args.name} already explained.[/red]"
                    )
                    return f"Attribute {args.name} already explained."

                locals_dict = {}

                with contextlib.redirect_stdout(io.StringIO()) as f:

                    exec(
                        f"{args.command}",
                        {
                            "pd": pd,
                            "df": df,
                            "np": np,
                            "signal": signal,
                            "tsfresh": tsfresh,
                            "df_features": df_features,
                            "df_time_series": df_time_series,
                        },
                        locals_dict,
                    )

                result = locals_dict.get("result", None)

                if result is None:
                    return ConsoleManager.console_error_print(
                        f"Error: Command did not set 'result' variable for attribute '{args.name}'.",
                        code=args.command,
                    )

                # Ensure alignment by position, not index
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    result = result.reset_index(drop=True).values

                if isinstance(result, np.ndarray):
                    if np.isnan(result).any() or np.isinf(result).any():
                        return ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains NaN or Inf values."
                        )
                else:
                    if pd.isnull(result) or result in [float("inf"), float("-inf")]:
                        return ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains NaN or Inf values."
                        )

                # Check that all values are numeric
                if isinstance(result, np.ndarray):
                    if not np.issubdtype(result.dtype, np.number):
                        return ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains non-numeric values."
                        )
                else:
                    # For scalar or Series/DataFrame converted to values
                    try:
                        _ = pd.Series(
                            result.flatten() if hasattr(result, "flatten") else result
                        ).astype(float)
                    except Exception:
                        return ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains non-numeric values."
                        )

                df_features[args.name] = result

                attribute_descriptions.append(
                    AttributeExplanation(
                        name=args.name,
                        argumentation=args.argumentation,
                        command=args.command,
                        status="Active",
                        added=f"Trail {step}" if step is not None else "N/A",
                    )
                )
                know_attributes.add(args.name)

                ConsoleManager.console_agent_logging(
                    self.__class__.__name__,
                    "append_new_attribute",
                    f"Name: [green]{args.name}[/green] \nArgumentation: {args.argumentation}",
                    code=args.command,
                    language="python",
                )

                self._tool_call_counter_success["append_new_attribute"] += 1
                return f"Attribute {args.name} explained, documented, and added to df_features."
            except Exception as e:

                return ConsoleManager.console_error_print(
                    f"Error updating attributes: {e}", code=args.command
                )

        @function_tool
        async def list_known_attributes_tool() -> str:
            """
            List all currently known attributes in the df_features dataframe.
            This tool can be used to check which attributes have already been explained and added to the df_features dataframe.

            Args:
                args (Any): No arguments are needed for this tool.

            Returns:
                str: A comma-separated string of known attribute names.
            """
            try:
                self._tool_call_counter["list_known_attributes_tool"] += 1

                known_attrs = ", ".join(sorted(set(df_features.columns) - {"id"}))
                ConsoleManager.console_agent_logging(
                    self.__class__.__name__,
                    "list_known_attributes_tool",
                    f"Known attributes ({len(set(df_features.columns) - {'id'})}) : {known_attrs}",
                )
                self._tool_call_counter_success["list_known_attributes_tool"] += 1
                return known_attrs
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error listing known attributes: {e}"
                )

        async def main():

            client = AsyncOpenAI(
                base_url=llm_cfg.endpoint,
                api_key=llm_cfg.api_key,
            )

            agent = Agent(
                model_settings=ModelSettings(
                    temperature=extractor_agent_cfg.temperature,
                    reasoning=extractor_agent_cfg.reasoning,
                ),
                name="Extractor Agent",
                instructions=f"""
                
                    # The Setup:
                    You are part of a team of agents working together to generate features with high predictive power.
                    The other agents in the team are:
                    - The Scientist Agent: {AgentRoles.SCIENTIST.value}
                    - The Tester Agent: {AgentRoles.TESTER.value}

                    You all work in a loop where the Scientist Agent defines a focus area for the investigation, you extract relevant attributes based on that focus, and the Tester Agent evaluates the quality of those attributes.


                    # Your Role:
                    You are a data aggregating assistant. You are provided with three pd.DataFrame objects:
                    - 'df': the main working dataframe containing raw data.
                    - 'df_features': The aggregated features from this and previous investigations.
                    - 'df_time_series': a copy of the 'df' dataframe with the 'datetime' column set as the index. Use this dataframe for time series analysis.
                    Use the provided 'generic_pandas_tool' tool to explore and analyze the data.
                    Your goal is to identify attributes that can be added to the 'df_features' dataframe using the 'append_new_attribute' tool.
                    The attributes you add should be relevant to the focus area provided by the Scientist Agent and should have high predictive power.
                    
                    # Your Workflow:
                    Follow these steps to achieve your goal:
                    1. Start by exploring the data using the 'generic_pandas_tool' tool to discover new attributes that can be added to the 'df_features' dataframe.
                    2. Use the 'append_new_attribute' tool to append, document and explain each new attribute you want to add to 'df_features'. For each attribute, provide:
                        - A clear and concise explanation of why the attribute is relevant and important.
                        - A detailed description of how the attribute is calculated, including any formulas or methods used.
                        - The exact pandas command used to calculate the attribute.
                    3. Repeat steps 1 and 2 until you have identified a sufficient amount of relevant attributes.


                    # The Context and Global Goal:
                    {self.cfg_experiment.context_and_goal}
        
                    # Tool Usage Guidelines:
                    You have the following tools at your disposal:
                    - 'generic_pandas_tool': You must use this tool to execute generic pandas commands to explore and analyze the data. 
                    - 'append_new_attribute': You must use this tool to document and explain all new attributes you want to add to the 'df_features' dataframe. These attributes should be tested using the generic_pandas_tool first.
                    - 'list_known_attributes_tool': You can use this tool to list all currently known attributes in the 'df_features' dataframe. This can help you avoid duplicating attributes.
                    - 'search_in_literature_tool': You can use this tool to search for relevant literature that can help you understand feature extraction techniques better.
                    
        
                    # Important Guidelines:
                    - The user will provide you with a focus area for the aggregation. This is a textual instruction and should guide your analysis.
                    - Use this focus area to guide your analysis and attribute creation.
                    - Only finish when you are certain that you have added a sufficient amount of relevant attributes to 'df_features'. Do not stop early. Use the 'list_known_attributes_tool' to check for existing attributes.
                    - Always think step by step and show your reasoning.
                    - Use the tools as often as needed.
                    - Categorical attributes should be one-hot encoded with flags when added to 'df_features'. E.x: 'has_diabetes' with values 0 and 1, or 'is_monday' with values 0 and 1.
                    - When creating new attributes, ensure they are relevant to the focus area provided by the Scientist
                    """,
                # Provide a final summary of the new attributes you have added to 'df_features' as a markdown table.
                model=OpenAIResponsesModel(
                    model=llm_cfg.model,
                    openai_client=client,
                ),
                tools=[
                    generic_pandas_tool,
                    append_new_attribute,
                    list_known_attributes_tool,
                    search_in_literature_tool,
                ],
            )

            # conversation = await client.responses.create(input="prompt")

            session = SQLiteSession(f"extractor_agent_session_step_{step or 0}")

            for _ in range(10):
                try:
                    res = await Runner().run(
                        agent,
                        f"System now wants to focus on: {focus}.",
                        max_turns=extractor_agent_cfg.max_iterations,
                        session=session,
                    )
                    # ConsoleManager.console_print(f"Agent Result: {res.final_output}")
                except Exception:
                    continue

        await main()

        @wandb_logging_wrapper
        def log_extractor_tool_calls():
            import wandb

            wandb.log(
                {
                    f"Extractor_tool_usage/{tool_name}": count
                    for tool_name, count in self._tool_call_counter.items()
                },
                step=step,
            )

            wandb.log(
                {
                    f"Extractor_tool_usage_success/{tool_name}": count
                    for tool_name, count in self._tool_call_counter_success.items()
                },
                step=step,
            )

        log_extractor_tool_calls()

        df_attribute_descriptions = pd.DataFrame(
            [attr.as_dict for attr in attribute_descriptions]
        )

        return df_features, df_attribute_descriptions


if __name__ == "__main__":

    import json
    from pathlib import Path

    ConsoleManager.console_rule("Testing Extractor Agent", color="red")

    df = pd.read_csv("/workspaces/BrainFlux/EHR_data/meds.csv")
    focus = "Medical history"

    agent = ExtractorAgent()
    df_features, attribute_descriptions = agent.generate_patient_attributes(df, focus)

    ConsoleManager.console_print(
        f"""Tool call counts: {'\n'.join(f"{k}: {v}" for k, v in dict(agent._tool_call_counter).items())}"""
    )

    ConsoleManager.console_print(
        f"Extracted {len(df_features.columns) - 1} patient attributes."
    )

    ConsoleManager.console_print(df_features)
    ConsoleManager.console_print(json.dumps(attribute_descriptions, indent=4))
