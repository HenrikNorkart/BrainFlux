# %%
import asyncio
from operator import index
from textwrap import dedent
import json
from logging import getLogger
import os


from requests import session

logger = getLogger(__name__)

from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAI
from rogueone.llm.clients import create_openai_client_for_vllm
from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIResponsesModel,
    ModelSettings,
    set_tracing_disabled,
    function_tool,
    WebSearchTool,
)

from agents.memory.sqlite_session import SQLiteSession
from agents.run import ModelInputData, CallModelData
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.utilities import GoogleSerperAPIWrapper

from rogueone.utils import embedding_cfg, llm_cfg, knowledge_agent_cfg
from rogueone.utils.console import ConsoleManager
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.retrieval_guards import assert_serper_live


# set_tracing_disabled(True)


def printer(func_name, command):
    ConsoleManager.console_agent_logging("KnowledgeAgent", func_name, command)


async def _run_search_agent(query: str) -> str:

    search = GoogleSerperAPIWrapper()

    @function_tool
    async def _web_search(query: str):
        """
        Perform a web search using the provided query.
        """

        try:
            search_results = search.run(query)
        except Exception as e:
            print(f"Error during web search: {e}")
            return "No results found due to an error."

        return search_results

    client = create_openai_client_for_vllm(
        agent_name="web_search",
        base_url=llm_cfg.endpoint,
        api_key=llm_cfg.api_key,
    )

    agent = Agent(
        model_settings=ModelSettings(
            temperature=knowledge_agent_cfg.temperature,
            reasoning=knowledge_agent_cfg.reasoning,
        ),
        name="Knowledge lookup assistant",
        # instructions="""Make one tool call and return the final answer.""",
        instructions=f"""
                    
            # The Setup:
            You are a knowledge lookup assistant that helps researchers find relevant information from the web.
            You are connected to the web via the web search tool.
            The user will provide you with queries related to their research.
                
            # Your Role and Tasks:
            Your goal is to answer the user's query in the best possible way using the information available from the web-search.
            You are operating as a RAG (Retrieval-Augmented Generation) agent.

            # Your Workflow:
            1. Break down the user's query into key concepts.
            2. For each key concept, use the web search tool to find relevant information from the web.
            3. Summarize the information retrieved from the web to answer the user's query.
            4. Provide a final response containing a relevant summary to the user based on the user's query.
                
            # Important Guidelines:
            - Always provide a final answer based on your search results, even if limited information is found.
            - You must only base your summaries on the information found using the web search tool.
            - Respond with "Unable to provide an explanation at this time." if no relevant information is found.
            """,
        model=OpenAIResponsesModel(
            model=llm_cfg.model,
            openai_client=client,
        ),
        tools=[_web_search],
    )

    def _strip_reasoning_items(data: CallModelData) -> ModelInputData:
        """Filter out 'reasoning' items from input to avoid vLLM 400 errors."""
        filtered = [
            item
            for item in data.model_data.input
            if not (isinstance(item, dict) and item.get("type") == "reasoning")
        ]
        return ModelInputData(input=filtered, instructions=data.model_data.instructions)

    session = SQLiteSession(f"web_search_agent")

    try:
        res = await Runner.run(
            agent,
            f"user's query: {query}",
            max_turns=knowledge_agent_cfg.max_iterations,
            session=session,
            run_config=RunConfig(call_model_input_filter=_strip_reasoning_items),
        )

        if not res.final_output or res.final_output == "":
            return "Unable to provide an explanation at this time."

        return res.final_output

    except Exception as e:
        print(f"Error occurred: {e}")
        return "Internal error."


class WebSearchAgent:
    def __init__(self, cfg: ExperimentConfig):
        self._cfg = cfg
        # This module used to overwrite SERPER_API_KEY at import time with a
        # hard-coded literal, so a key supplied via .secret.env had no effect.
        # The key now comes from the environment only, and its absence is a
        # launch-time failure rather than a silent per-query error string.
        assert_serper_live()

    async def explain_query(self, query: str) -> str:
        ans = await _run_search_agent(query)
        return ans
