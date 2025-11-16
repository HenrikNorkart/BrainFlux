# %%
import asyncio
from operator import index
from textwrap import dedent
import json
from logging import getLogger

from requests import session

logger = getLogger(__name__)

from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAI
from agents import (
    Agent,
    Runner,
    OpenAIResponsesModel,
    ModelSettings,
    set_tracing_disabled,
    function_tool,
)
from agents.memory.sqlite_session import SQLiteSession
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from brainflux.utils import embedding_cfg, llm_cfg, knowledge_agent_cfg
from brainflux.utils.console import ConsoleManager
from brainflux.utils.config import ExperimentConfig


# set_tracing_disabled(True)


def printer(func_name, command):
    ConsoleManager.console_agent_logging("KnowledgeAgent", func_name, command)


class SearchArgs(BaseModel):
    query: str = Field(..., description="The search query.")
    k: int = Field(5, description="The number of similar documents to return.")


async def _run_explanation_agent(query: str, db: Chroma) -> str:
    client = AsyncOpenAI(
        base_url=llm_cfg.endpoint,
        api_key=llm_cfg.api_key,
    )

    @function_tool
    async def _similarity_search(args: SearchArgs):
        """
        Search for similar chunks in the vector DB.
        """

        # printer("_similarity_search", args.query)
        # print(f"Performing similarity search for query: {args.query} with k={args.k}")

        res = await db.asimilarity_search_with_relevance_scores(args.query, k=args.k)

        # print(f"Similarity search returned {len(res)} results.")
        # print(f"Results: {res[0]}")

        if not res:
            return "No data found."
        # Similarity score: {score}\n

        out = "\n\n".join([f"Content: {doc.page_content}" for doc, score in res])

        # print(f"Similarity search output: {out}")
        return out

    agent = Agent(
        model_settings=ModelSettings(
            temperature=knowledge_agent_cfg.temperature,
            reasoning=knowledge_agent_cfg.reasoning,
        ),
        name="Knowledge lookup assistant",
        # instructions="""Make one tool call and return the final answer.""",
        instructions=f"""
                    
            # The Setup:
            You are a knowledge lookup assistant that helps researchers find relevant information from a database.
            You are connected to a vector database that contains relevant literature and documents.
            By using the 'similarity_search' tool, you can retrieve information that semantically similar to the user's query.
                
            # Your Role and Tasks:
            Your goal is to answer the user's query in the best possible way using the information available in the vector database.
            You are operating as a RAG (Retrieval-Augmented Generation) agent.

            # Your Workflow:
            1. Break down the user's query into key concepts.
            2. For each key concept, use the 'similarity_search' tool to find relevant information from the database.
            3. Summarize the information retrieved from the database to answer the user's query.
            4. Provide a final response containing a relevant summary to the user based on the user's query.
                
            # Important Guidelines:
            - Always provide a final answer based on your search results, even if limited information is found.
            - You must only base your summaries on the information found using the 'similarity_search' tool.
            - Respond with "Unable to provide an explanation at this time." if no relevant information is found.
            """,
        model=OpenAIResponsesModel(
            model=llm_cfg.model,
            openai_client=client,
        ),
        tools=[_similarity_search],
    )

    session = SQLiteSession(f"knowledge_agent_session_db_{db._collection_name}")

    try:
        res = await Runner.run(
            agent,
            f"user's query: {query}",
            max_turns=knowledge_agent_cfg.max_iterations,
            session=session,
        )

        if not res.final_output or res.final_output == "":
            return "Unable to provide an explanation at this time."

        return res.final_output

    except Exception as e:
        print(f"Error occurred: {e}")
        return "Internal error."


class KnowledgeAgent:
    def __init__(self, cfg: ExperimentConfig, collection_name: str | None = None):
        self._cfg = cfg

        self._vector_db = Chroma(
            collection_name=collection_name or self._cfg.knowledge_db_collection_name,
            persist_directory=str("/workspaces/BrainFlux/chroma_db_medical_knowledge"),
            embedding_function=OpenAIEmbeddings(
                base_url=embedding_cfg.endpoint,
                api_key=embedding_cfg.api_key,
                model=embedding_cfg.model,
                tiktoken_enabled=True,
                chunk_size=embedding_cfg.chunk_size,
            ),
        )

    async def explain_query(self, query: str) -> str:
        ans = await _run_explanation_agent(query, db=self._vector_db)
        return ans

    def upload_pdf(self, pdf_path: str) -> int:
        """
        Loads a PDF, splits it into chunks, and stores it in the vector DB.
        Returns the number of chunks added.
        """
        print(f"Uploading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for i in range(0, len(documents), 5000):
            self._vector_db.add_documents(documents[i : min(i + 5000, len(documents))])

        return len(documents)

    def upload_json(self, json_path: str) -> int:
        """
        Loads a JSON file, splits it into chunks, and stores it in the vector DB.
        Returns the number of chunks added.
        """
        print(f"Uploading JSON: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        title = data.get("title", "Unknown Title")
        print(f"Title: {title}")
        sections = data.get("content", [])
        print(f"Number of sections: {len(sections)}")

        # Assuming the JSON structure is a list of text entries
        documents = [
            Document(
                page_content=dedent(
                    f"""
                Title: {title}
                Section: {entry.get("section_title", "Unknown Section")}
                {entry.get("content", "")}
                """
                ).strip(),
            )
            for entry in sections
        ]

        for i in range(0, len(documents), 5000):
            self._vector_db.add_documents(documents[i : min(i + 5000, len(documents))])

        return len(documents)

    def num_documents(self) -> int:
        """Return the number of documents in the vector store."""
        return self._vector_db._collection.count()  # type: ignore

    # def explain_query(query: str) -> str:


#     return asyncio.run(_run_explanation_agent(query))


if __name__ == "__main__":

    from pathlib import Path
    import asyncio

    base_path = Path("/workspaces/BrainFlux/tasks/vehicle/literatur/json")

    cfg = ExperimentConfig.from_yaml("/workspaces/BrainFlux/tasks/vehicle/config.yml")

    agent = KnowledgeAgent(cfg, collection_name=cfg.knowledge_db_collection_name)

    # for json_file in base_path.glob("**/*.json"):
    #     num_chunks = agent.upload_json(str(json_file))
    #     print(f"Uploaded {num_chunks} chunks from {json_file}\n\n")

    query = "What are common practices for feature extraction?"
    print(f"Top 3 results for query '{query}':")

    print(f"Number of documents in the vector DB: {agent.num_documents()}")

    # query = "Are there any medications that pops out in the literature?"

    async def main():
        res = await agent.explain_query(query)
        print(f"Query: {query}")
        print(f"Summary: {res}")

    asyncio.run(main())
