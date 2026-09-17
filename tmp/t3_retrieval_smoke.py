"""T3 (HQ #126) -- positive proof that both retrieval channels are live.

Run inside the BrainFlux container, with the vLLM LLM + embedding servers up:

    python3 -u /workspace/tmp/t3_retrieval_smoke.py /workspace/phantom_menace/config_24h.yml

It prints, with timestamps:

  1. the medical-knowledge Chroma store's path, collection and embedding count;
  2. a real passage returned by a similarity search against that store;
  3. a raw Serper result (proves key + network);
  4. the full WebSearchAgent.explain_query() path -- the same call the
     ScientistAgent's ``search_tool`` wraps.

Any dead channel now raises DeadRetrievalChannel at construction time instead of
degrading silently, so a clean exit is itself part of the proof.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone


def ts() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def banner(msg: str) -> None:
    print(f"\n[{ts()}] ===== {msg} =====", flush=True)


async def main(config_path: str) -> int:
    from rogueone.utils.config import ExperimentConfig
    from rogueone.llm.agents.knowledge import KnowledgeAgent
    from rogueone.llm.agents.web_search import WebSearchAgent

    cfg = ExperimentConfig.from_yaml(config_path)

    banner("CONFIG")
    print(f"config                     : {config_path}")
    print(f"experiment_name            : {cfg.experiment_name}")
    print(f"knowledge_db_path          : {cfg.knowledge_db_path}")
    print(f"knowledge_db_collection    : {cfg.knowledge_db_collection_name}")

    banner("PROOF 1/4 -- knowledge store embedding count")
    # KnowledgeAgent.__init__ runs assert_knowledge_store_live(), which raises if
    # the store is empty. Reaching the next line already proves count > 0.
    agent = KnowledgeAgent(cfg)
    count = agent.num_documents()
    print(f"embedding count            : {count}")
    if count <= 0:
        print("FAIL: store is empty")
        return 1

    banner("PROOF 2/4 -- similarity search returns a real passage")
    query = "What vital-sign patterns in the first 24 hours predict outcome after cardiac arrest?"
    docs = agent._vector_db.similarity_search(query, k=3)
    print(f"query                      : {query}")
    print(f"documents returned         : {len(docs)}")
    if not docs:
        print("FAIL: similarity search returned nothing")
        return 1
    for i, d in enumerate(docs):
        print(f"\n--- passage {i + 1} ({len(d.page_content)} chars) ---")
        print(d.page_content[:800])

    banner("PROOF 3/4 -- raw Serper call")
    from langchain_community.utilities import GoogleSerperAPIWrapper

    key = os.getenv("SERPER_API_KEY", "")
    print(f"SERPER_API_KEY             : {'set (…' + key[-4:] + ')' if key else 'MISSING'}")
    raw = GoogleSerperAPIWrapper().run(
        "post-cardiac-arrest syndrome early prognostication vital signs"
    )
    print(f"raw serper result ({len(raw)} chars):")
    print(raw[:800])
    if not raw.strip():
        print("FAIL: empty serper result")
        return 1

    banner("PROOF 4/4 -- WebSearchAgent.explain_query (the search_tool path)")
    ws = WebSearchAgent(cfg=cfg)
    wq = "Which early haemodynamic variables are associated with neurological outcome after in-hospital cardiac arrest?"
    ans = await ws.explain_query(wq)
    print(f"query                      : {wq}")
    print(f"answer ({len(ans)} chars):")
    print(ans[:2000])
    if (not ans.strip()) or ans.strip() in (
        "Internal error.",
        "Unable to provide an explanation at this time.",
    ):
        print("FAIL: web search path returned an error/empty sentinel")
        return 1

    banner("ALL FOUR PROOFS PASSED")
    return 0


if __name__ == "__main__":
    cfg_path = (
        sys.argv[1] if len(sys.argv) > 1 else "/workspace/phantom_menace/config_24h.yml"
    )
    sys.exit(asyncio.run(main(cfg_path)))
