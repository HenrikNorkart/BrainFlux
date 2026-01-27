# %%
from logging import getLogger
from collections import defaultdict

import pandas as pd
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIResponsesModel,
    ModelSettings,
    function_tool,
)

from agents.memory.sqlite_session import SQLiteSession

from rogueone.utils import (
    llm_cfg,
    scientist_agent_cfg,
)
from rogueone.utils.config import ExperimentConfig
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
        self.web_search_agent = WebSearchAgent(cfg=self.cfg_experiment)

    async def summarize_experiments(
        self,
        df_test_results: pd.DataFrame,
        df_attribute_explanations: pd.DataFrame,
    ) -> str:

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)

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
                name="Summarizer Agent",
                instructions=f"""
                    The following files contains the output from an optimization loop. 
                    The task is to discover attributes from patients EHR that would indicate that their EEG readings are not suitable for determining if they will survive. 
                    The first file "df_test_pool_final" contains the metrics and summary of every trail. 
                    The second document "df_attributes_final (3)" contains definitions for each attribute. 
                    Although some attributes are set to "Pruned", this is not important and you should NOT pay attention to this.
                    please look at all nine trials and describe what are the common ground between the strongest parameters.
                    
                    Use the "attribute_lookup_tool" to get explanations and definitions for specific attributes.
                    Your final output should be a concise summary that includes:
                    1. An overview of the most impactful attributes identified across the tests grouped by common ground (e.g., attributes related to heart rate variability, attributes related to medication history, etc.). 
                        Keep the summary brief and only include 5 top features.  
                    2. Use the "search_tool" to find supporting evidence from scientific literature that validates the importance of these 5 attributes in predicting abnormal EEG patterns. 
                        Are the findings consistent with established knowledge in the field?
                    
                    Take time to think through the problem step by step and provide a well-structured summary.
                    USe the "attribute_lookup_tool" to gain technical definition of each attribute in the summary. This includes the logical reasoning behind why these attributes are important and the code implementation.
                                        
                    """,
                # f"""
                #     # The Setup:
                #     You are a part of a team of AI agents working together to find insightful attributes from time-series data.
                #     The team has conducted several experimental tests and trails, which have yielded a variety of results.
                #     These results are a large collection of new attributes, metrics, and observations derived from the tests.
                #     # Your Role and Tasks:
                #     You are a highly intelligent summarizer agent tasked with analyzing and summarizing experimental test results.
                #     Your primary objective is to read the reports from each trail and identify the most impactful attributes that contribute to the outcomes.
                #     You should then look up these attributes to find common ground and trends across different tests.
                #     # Your Workflow:
                #     You will utilize the following tools to assist you in your analysis:
                #     1. Use the 'trail_lookup_tool' to retrieve specific test results based on trial IDs.
                #     2. Use the 'attribute_lookup_tool' to get detailed explanations of specific attributes.
                #     3. Identify trends and common ground between the most impactful attributes across different tests.
                #     4. Use the 'search_tool' to gather additional information that may help contextualize the findings.
                #     5. Synthesize the information gathered into a coherent and concise summary consisting of two parts:
                #         a. An overview of the most impactful attributes identified across the tests and their common ground.
                #         b. An analysis of how these findings relate to established knowledge in the field, supported by evidence from your searches.
                #     # The Context and Global Goal of the team of agents:
                #     The overall goal is to use data from EHR-logs to generate a set of attributes that can help identify patients that are likely to have abnormal EEG patterns, thus the survival of these patients can not be determined from their EEG readings.
                #     The other agents in the team have already contributed by generating new attributes, running tests, and providing explanations for various attributes.
                #     The goal is to highlight the most important attributes that can help in identifying these patients effectively.
                #     # Important Guidelines:
                #     - Be concise and focus on the most significant findings.
                #     - Some attributes are set to "pruned"; These attributes are still important and should be considered in your analysis.
                #     - Provide evidence and reasoning for your conclusions.
                #     - Ensure that your summary is well-structured and easy to understand.
                #     - we only care about the feature importance and attribute explanations, not the other metrics.
                #     """,
                model=OpenAIResponsesModel(
                    model=llm_cfg.model,
                    openai_client=client,
                ),
                tools=[
                    search_tool,
                    attribute_lookup_tool,
                    # trail_lookup_tool,
                ],
            )

            trails = df_test_results["id"].unique().tolist()

            prompt = f"Trails: {df_test_results.to_dict(orient="records")}"
            # prompt = f"Please summarize the test results across all trials: {', '.join(trails)}."

            session = SQLiteSession(f"Exploration_Session")

            for _ in range(100):
                try:
                    focus = await Runner().run(
                        agent,
                        prompt,
                        session=session,
                        max_turns=500,
                    )
                    if len(focus.final_output) > 10:
                        return focus.final_output
                except Exception:
                    continue
            return "Unable to determine focus."

        return await main()


if __name__ == "__main__":
    from asyncio import run
    from pathlib import Path

    from agents import set_tracing_disabled

    from rogueone.utils.config import ExperimentConfig

    set_tracing_disabled(True)

    cfg = ExperimentConfig.from_yaml("/workspaces/BrainFlux/phantom_menace/config.yml")

    test_pool = pd.read_csv(
        "/workspaces/BrainFlux/phantom_menace/output/df_test_pool_final.csv"
    )
    attribute_explanations = pd.read_csv(
        "/workspaces/BrainFlux/phantom_menace/output/df_attributes_explanations_final.csv"
    )

    agent = ScientistAgent(cfg=cfg)
    focus = run(
        agent.summarize_experiments(
            df_test_results=test_pool,
            df_attribute_explanations=attribute_explanations,
        )
    )

    print("-" * 80)

    print("Done!")

    Path("/workspaces/BrainFlux/dummy.md").write_text(focus)
