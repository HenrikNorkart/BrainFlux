"""#126 criterion 2 -- positive proof the clinical-literature store is LIVE.

A store that merely opens without error does not count; that is exactly the failure
mode that went invisible for a whole run. So this asserts two things:

  1. the collection the config names holds a non-zero embedding count, and
  2. a real semantic query returns an actual passage, printed verbatim.

Run inside the container, with the embedding server already up.
"""
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ws = Path("/workspace")
for f in (".env", ".dev.env", ".secret.env"):
    p = ws / f
    if p.exists():
        load_dotenv(p, override=(f == ".dev.env"))

cfg = yaml.safe_load(open(ws / "phantom_menace" / "config.yml"))
path = cfg["knowledge_db_path"]
collection = cfg["knowledge_db_collection_name"]
print(f"config knowledge_db_path       : {path}")
print(f"config knowledge_db_collection : {collection}")
print(f"path exists                    : {Path(path).exists()}")

from langchain_chroma import Chroma  # noqa: E402
from langchain_openai.embeddings import OpenAIEmbeddings  # noqa: E402

emb = OpenAIEmbeddings(
    base_url=f"http://localhost:{os.getenv('EMBEDD_PORT')}/v1",
    api_key=os.getenv("EMBEDD_API_KEY"),
    model=os.getenv("EMBEDD_MODEL"),
    tiktoken_enabled=True,
)
db = Chroma(collection_name=collection, persist_directory=path, embedding_function=emb)

count = db._collection.count()
print(f"\n[1] embedding count            : {count}")
if count == 0:
    print("FAIL: store is empty")
    sys.exit(1)

queries = [
    "suppression ratio EEG prognosis after cardiac arrest",
    "sedation and its effect on EEG interpretation in intensive care",
]
ok = 0
for q in queries:
    docs = db.similarity_search(q, k=2)
    print(f"\n[2] query: {q!r} -> {len(docs)} passages")
    for d in docs[:1]:
        text = d.page_content.strip().replace("\n", " ")
        print(f"    PASSAGE ({len(d.page_content)} chars): {text[:400]}")
        if len(text) > 50:
            ok += 1

print(f"\nqueries returning a real passage: {ok}/{len(queries)}")
sys.exit(0 if ok == len(queries) else 1)
