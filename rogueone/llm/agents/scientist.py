# %%
import io
import contextlib
import asyncio
from typing_extensions import Any
from logging import getLogger
from collections import defaultdict

from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIResponsesModel,
    ModelSettings,
    set_tracing_disabled,
    function_tool,
)

from agents.memory.sqlite_session import SQLiteSession
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from rogueone.utils import (
    llm_cfg,
    scientist_agent_cfg,
)
from rogueone.utils.config import ExperimentConfig
from rogueone.llm.agents.knowledge import KnowledgeAgent
from rogueone.llm.agents.web_search import WebSearchAgent
from rogueone.utils.console import ConsoleManager
from rogueone.dataclasses import AttributeExplanation
from rogueone.utils.wandb_utils import wandb_logging_wrapper
from rogueone.llm.prompts.agent_roles import AgentRoles


# set_tracing_disabled(True)
logger = getLogger(__name__)


class ScientistAgent:

    def __init__(self, cfg: ExperimentConfig):
        self.cfg_experiment = cfg
        self.knowledge_agent = KnowledgeAgent(cfg=self.cfg_experiment)
        self.web_search_agent = WebSearchAgent(cfg=self.cfg_experiment)

        self._note_book = ""

    async def determine_focus(
        self,
        df_raw_data: pd.DataFrame,
        df_test_results: pd.DataFrame | None,
        df_attribute_explanations: pd.DataFrame,
        focus_history: list[str],
        index: int,
        max_index: int,
    ) -> str:
        """Determine the focus of the investigation based on test results and attribute explanations.

        Args:
            df_test_results (pd.DataFrame): The DataFrame containing test results.
            attribute_explanations (dict): A dictionary mapping attribute names to their explanations.
        Returns:
            str: The determined focus of the investigation.
        """

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)

        _df_test_results = (
            df_test_results.copy() if df_test_results is not None else pd.DataFrame()
        )
        _df_base_data = df_raw_data.copy()

        self.web_search_agent

        @function_tool
        async def search_tool(query: str) -> str:
            """
            Search the web for relevant information to expand your knowledge.
            Use this tool to find supporting evidence or context.
            Use this tool to get a better understanding of relevant concepts.

            Args:
                query (str): The search query.
            Returns:
                str: A summary of the explanation.
            """
            try:
                self._tool_call_counter["search_tool"] += 1
                explanation = await self.web_search_agent.explain_query(query)

                # Normalize and check for empty responses
                explanation = explanation if explanation is not None else ""
                if str(explanation).strip() == "":
                    return ConsoleManager.console_error_print(
                        f"No explanation found for query: {query}"
                    )

                max_len = min(1000, len(explanation))

                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "search_tool",
                    query,
                    post_message=explanation[:max_len]
                    + ("..." if len(explanation) > max_len else ""),
                )
                self._tool_call_counter_success["search_tool"] += 1
                return explanation
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error during literature search: {e}"
                )

        # @function_tool
        # async def search_in_literature_tool(query: str) -> str:
        #     """
        #     Search up and get a summary from relevant literature to explain the domain of the data.
        #     Use this tool to find supporting evidence or context.
        #     Use this tool to get a better understanding of relevant concepts.

        #     Args:
        #         query (str): The search query.
        #     Returns:
        #         str: A summary of the explanation.
        #     """
        #     try:
        #         self._tool_call_counter["search_in_literature_tool"] += 1
        #         explanation = await self.knowledge_agent.explain_query(query)

        #         # Normalize and check for empty responses
        #         explanation = explanation if explanation is not None else ""
        #         if str(explanation).strip() == "":
        #             return ConsoleManager.console_error_print(
        #                 f"No explanation found for query: {query}"
        #             )

        #         ConsoleManager.console_agent_logging(
        #             "ScientistAgent", "search_in_literature_tool", query
        #         )
        #         return explanation
        #     except Exception as e:
        #         return ConsoleManager.console_error_print(
        #             f"Error during literature search: {e}"
        #         )

        @function_tool
        async def generic_pandas_tool(command: str) -> Any:
            """
            Execute a generic pandas command on the test results dataframe 'df_test_results' or the base data dataframe 'df_raw_data'.
            The dataframe 'df_test_results' is organized as follows:
            - Each row represents a the results from one round of investigations.
            - The columns include:
                - The first column is the trail number. This is just an identifier with the form "Trail<trail_number>". Newer trails have higher numbers.
                - The next columns are the classification metrics: accuracy, precision, recall, f1 scores and AUROC. These metrics indicate how well the attributes predict the given task.
                - The final column is a rapport from the tester agent that describes the overall assessment of the attributes.

            The dataframe 'df_raw_data' contains the original attributes used for aggregating.

            Use this tool to explore the dataframe and generate insights into the test results.
            This tool can be used to filter, aggregate, or view the data.

            The command should always:
            1. Use 'result' as the variable name for the output of the command.
               For example, if you want to get the mean accuracy, you should use:
               result = df_test_results['accuracy'].mean()
            2. Ensure that the command is a valid pandas command that can be executed in the context of the dataframe 'df_test_results' or 'df_raw_data'.
               You can use any pandas function or method that is applicable to dataframes.

            The tool will execute the following code:
                ```python
                locals_dict = {}

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    exec(
                        f"{command}",
                        {
                            "pd": pd,
                            "np": np,
                            "df_test_results": _df_test_results,
                            "df_raw_data": _df_base_data,
                        },
                        locals_dict,
                    )

                result = locals_dict.get("result")
                return result
                ```

            Args:
                command (str): The pandas command to execute.

            Returns:
                Any: The result of the pandas command.
            """
            try:
                self._tool_call_counter["generic_pandas_tool"] += 1
                locals_dict = {}

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    exec(
                        f"{command}",
                        {
                            "pd": pd,
                            "np": np,
                            "df_test_results": _df_test_results,
                            "df_raw_data": _df_base_data,
                        },
                        locals_dict,
                    )

                result = locals_dict.get("result", None)
                if result is None:
                    return ConsoleManager.console_error_print(
                        f"Error: Command did not set 'result' variable or 'result' was set to None! Command: {command}"
                    )

                ConsoleManager.console_agent_logging(
                    "ScientistAgent", "generic_pandas_tool", code=command
                )
                self._tool_call_counter_success["generic_pandas_tool"] += 1
                return result
            except Exception as e:
                return ConsoleManager.console_error_print(
                    f"Error executing command: {e}", code=command
                )

        @function_tool
        async def list_all_attributes() -> str:
            """
            List all attribute names available in the dataframe 'df'.
            These names can be used as input for the 'attribute_lookup_tool'.

            Returns:
                str: A comma-separated list of attribute names.
            """

            try:
                self._tool_call_counter["list_all_attributes"] += 1
                if len(df_attribute_explanations) == 0:
                    res = "No attributes available. This indicates that this is the first round of investigation."

                else:
                    res = df_attribute_explanations["Attribute"].to_list()
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "list_all_attributes",
                    f"List all attributes ({len(df_attribute_explanations)} total)",
                )

                self._tool_call_counter_success["list_all_attributes"] += 1
                return res
            except Exception as e:
                ConsoleManager.console_error_print(f"Error listing attributes: {e}")

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
                    "ScientistAgent",
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
        async def take_a_note(note: str) -> str:
            """
            Take a note and add it to the notebook.
            This can be used to keep track of important information during the investigation and to pass information on to next iteration.

            Args:
                note (str): The note to be added.

            Returns:
                str: A confirmation message.
            """

            try:
                self._tool_call_counter["take_a_note"] += 1
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "take_a_note",
                    f"Taking note: {note}",
                )

                self._note_book += f"Trail {index}:\n{note}\n"

                self._tool_call_counter_success["take_a_note"] += 1
                return "Note added successfully."
            except Exception as e:
                return ConsoleManager.console_error_print(f"Error taking note: {e}")

        @function_tool
        async def read_notes() -> str:
            """
            Read all notes from the notebook.

            Returns:
                str: The contents of the notebook.
            """

            try:
                self._tool_call_counter["read_notes"] += 1
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "read_notes",
                    "Reading all notes",
                )

                self._tool_call_counter_success["read_notes"] += 1
                return self._note_book
            except Exception as e:
                return ConsoleManager.console_error_print(f"Error reading notes: {e}")

        async def main():

            client = AsyncOpenAI(
                base_url=llm_cfg.endpoint,
                api_key=llm_cfg.api_key,
            )

            agent = Agent(
                model_settings=ModelSettings(
                    temperature=scientist_agent_cfg.temperature,
                    reasoning=scientist_agent_cfg.reasoning,
                ),
                name="Researcher Agent",
                instructions=f"""

                    # The Setup:
                    You are part of a team of agents working together to generate features with high predictive power.
                    The other agents in the team are:
                    - The Extractor Agent: {AgentRoles.EXTRACTOR.value}
                    - The Tester Agent: {AgentRoles.TESTER.value}

                    You all work in a loop where the Extractor Agent extracts attributes, the Tester Agent tests them, and you analyze the results to determine the focus of the investigation. You then generate new hypotheses based on this focus.


                    # Your Role and Tasks:
                    You are a researcher agent that is tasked with leading the investigative work. 
                    Your primary responsibility is to determine the focus of the investigation based on the test results provided by the tester agent and the explanations of the attributes extracted by the extractor agent.
                    Your task is to analyze the test results, the attribute explanations, the focus history, and compare with existing information from the web to determine the focus area of the investigation. Find the focus areaa that will lead to the most useful hypotheses.
                    Use the tools available to you to explore the data, gather information, and generate insights.


                    # Your Workflow:
                    Use the following workflow to determine the focus:
                    0. Review the notebook to gather any thoughts from previous iterations.
                    1. Analyze the test results to identify the attributes that are most predictive. Use the rapports from the tester agent to determine what attributes are most useful.
                    2. Review the explanations of the attributes to understand their significance and relevance to the investigation.
                    3. Consider the focus history to avoid repeating previous focuses and to build on past insights.
                    4. Use the tools available to you to gather additional information and context. This may include searching the web, exploring the dataframe, and looking up attribute explanations.
                    5. Synthesize the information gathered from the previous steps to refine the focus of the investigation. Make sure the focus area is within the scope of the available data found in 'df_raw_data'.
                    6. Choose a focus that should be used for further attribute extraction. Express whether the focus should be exploratory (broad focus) or exploitative (narrow focus).
                    7. Use the notebook to keep track of important information and to pass information on to the next iteration.


                    # The Context and Global Goal:
                    {self.cfg_experiment.context_and_goal}
                    
                        
                    # Important Guidelines:
                    - Use the 'search_tool' actively to find relevant information from the web to support your focus area.
                    - Always think step-by-step and explain your reasoning.
                    - Your strategy is to explore hints within the best attributes. For example, if "age" is a good attribute, then you might want to explore "age" further by looking into attributes that are derived from age.
                    - If there are many rounds left, you can afford to be more exploratory. If there are only a few rounds left, then you should focus on the most promising attributes and avoid exploring new areas.
                    - Your final answer should be a concise statement of the focus of the investigation and a brief explanation of why it was chosen. Keep it to a few sentences.
                    - Only respond when you are certain of the focus. If you are unsure, use the tools to gather more information. If you are unable to determine a focus, then use a wildcard focus such as "Explore general age related features".
                    - For the first round of investigation it is smart to look into the performance of the raw attributes. Thus, a good focus for the first round is often exploratory and looking at only the raw attributes without any transformations or derived features.

                    # Extractor Agent Constraints:
                    - The Extractor Agent can only use the 'Pandas' and 'Numpy' libraries to generate new attributes. Thus, the focus should be on attributes that can be derived using these libraries.
                    - For time series data the Extractor Agent can use the 'tsfresh' and 'scipy.signal' libraries to generate new attributes. These libraries offer more complex features that can be useful for extractions with an exploitative "focus area". Thus, the focus can also be on time series attributes that can be derived using these libraries.
                    - Use the lookup tool if you need to understand more information about the libraries available for the Extractor Agent.
                    """,
                model=OpenAIResponsesModel(
                    model=llm_cfg.model,
                    openai_client=client,
                ),
                tools=[
                    generic_pandas_tool,
                    # search_in_literature_tool,
                    search_tool,
                    attribute_lookup_tool,
                    list_all_attributes,
                    take_a_note,
                    read_notes,
                ],
            )

            if len(focus_history) > 0:
                prompt = f"Round {index}/{max_index}: Determine the focus for the next round of attribute extraction based the focus history: {focus_history}."
            else:
                prompt = f"Round 1: Determine the focus for the next round of attribute extraction. This is the first round of investigation, thus there is no focus history or extracted attributes."

            session = SQLiteSession(f"scientist_agent_session_step_{index or 0}")

            for _ in range(100):
                try:
                    focus = await Runner().run(
                        agent,
                        prompt,
                        session=session,
                        max_turns=scientist_agent_cfg.max_iterations,
                    )
                    if len(focus.final_output) > 10:
                        return focus.final_output
                except Exception:
                    continue
            return "Unable to determine focus."

        focus = await main()

        @wandb_logging_wrapper
        def log_focus_to_wandb():
            import wandb

            # Log the focus determined by the ScientistAgent
            wandb.log({f"Research_focus": focus}, step=index)

            # Log the tool usage counts
            wandb.log(
                {
                    f"Scientist_tool_usage/{tool_name}": count
                    for tool_name, count in self._tool_call_counter.items()
                },
                step=index,
            )

            wandb.log(
                {
                    f"Scientist_tool_usage_success/{tool_name}": count
                    for tool_name, count in self._tool_call_counter_success.items()
                },
                step=index,
            )

        log_focus_to_wandb()

        return focus


if __name__ == "__main__":

    import json
    from pathlib import Path
    from rogueone.utils.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(
        "/workspaces/BrainFlux/tasks/suppression_rato/config.yml"
    )

    patient_attributes = pd.read_csv("/workspaces/BrainFlux/tmp/df_patients.csv")
    attribute_explanations = json.loads(
        Path("/workspaces/BrainFlux/tmp/attribute_descriptions.json").read_text()
    )

    agent = ScientistAgent(cfg=cfg)
    focus = agent.determine_focus(
        df_raw_data=patient_attributes,
        df_test_results=pd.DataFrame(),
        df_attribute_explanations=attribute_explanations,
        focus_history=[],
        index=0,
        max_index=5,
    )

    print("-" * 80)

    print(focus)
