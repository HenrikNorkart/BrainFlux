"""Startup guards for the two retrieval channels.

The 24 h discovery run (WandB ``cardiac_arrest_24h`` / ``869q3b4z``) completed a
290-minute loop with *both* retrieval channels dead and said nothing about it:

* the medical-knowledge Chroma store was opened from a hard-coded path that did
  not exist on the run host, so Chroma silently auto-created an empty store;
* the Serper key was absent, so every web search returned an error string that
  the agent happily summarised as "no information found".

Both failures are invisible from the outside: the loop still runs, still writes
attributes, still logs to WandB. These guards turn each of them into a loud
failure at construction time instead.

Set ``ROGUE_ONE_ALLOW_DEAD_RETRIEVAL=1`` to downgrade the guards to warnings --
only do that when a dead channel is the thing being measured (e.g. a deliberate
no-retrieval ablation), and record it in the run config.
"""

import os
from logging import getLogger

logger = getLogger(__name__)


class DeadRetrievalChannel(RuntimeError):
    """Raised at launch when a retrieval channel cannot serve real content."""


def _allow_dead() -> bool:
    return os.getenv("ROGUE_ONE_ALLOW_DEAD_RETRIEVAL", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _fail(message: str) -> None:
    if _allow_dead():
        logger.warning(
            "ROGUE_ONE_ALLOW_DEAD_RETRIEVAL is set -- continuing with a dead "
            "retrieval channel: %s",
            message,
        )
        print(f"[retrieval-guard][WARN] {message}")
        return
    raise DeadRetrievalChannel(message)


def assert_knowledge_store_live(vector_db, persist_directory, collection_name) -> int:
    """Assert the knowledge Chroma store actually holds embeddings.

    Returns the embedding count so the caller can log it. A store that merely
    opens without error does not count -- that is exactly the failure mode that
    went invisible for a whole run.
    """
    try:
        count = int(vector_db._collection.count())  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        _fail(
            f"Could not count embeddings in knowledge store "
            f"'{collection_name}' at {persist_directory}: {exc}"
        )
        return -1

    print(
        f"[retrieval-guard] knowledge store '{collection_name}' at "
        f"{persist_directory}: {count} embeddings"
    )
    logger.info(
        "knowledge store %s at %s holds %d embeddings",
        collection_name,
        persist_directory,
        count,
    )

    if count == 0:
        _fail(
            f"Knowledge store '{collection_name}' at {persist_directory} is EMPTY "
            f"(0 embeddings). Chroma auto-creates an empty store when the path or "
            f"collection name is wrong, so this is almost certainly a "
            f"knowledge_db_path / knowledge_db_collection_name mismatch rather "
            f"than an intentionally empty corpus."
        )

    return count


def assert_serper_key_present() -> str:
    """Assert a Serper API key is configured, and return it."""
    key = (os.getenv("SERPER_API_KEY") or "").strip()
    if not key:
        _fail(
            "SERPER_API_KEY is not set. The web-search channel will return an "
            "error string for every query, which the agent summarises as 'no "
            "information found' -- indistinguishable from a real negative."
        )
        return ""
    print(f"[retrieval-guard] SERPER_API_KEY present (…{key[-4:]})")
    return key
