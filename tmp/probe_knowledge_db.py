"""#120 criterion 1 -- was the medical-knowledge DB ever populated?

If the WebSearchAgent's retrieval ever succeeded, the knowledge collection should hold
documents. An empty collection is evidence that search never worked for the published
runs either, which would retroactively re-label them as No Search.
"""
import sqlite3
import sys

path = sys.argv[1]
c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
print("tables:", tabs)
for t in tabs:
    try:
        n = c.execute(f"select count(*) from {t}").fetchone()[0]
        print(f"  {t:45s} {n}")
    except Exception as exc:
        print(f"  {t:45s} ERR {exc}")

for t in ("collections", "segments"):
    if t in tabs:
        print(f"\n--- {t} ---")
        for r in c.execute(f"select * from {t} limit 10"):
            print("   ", r)

for t in ("embedding_fulltext_search_content", "embedding_metadata"):
    if t in tabs:
        print(f"\n--- sample {t} ---")
        for r in c.execute(f"select * from {t} limit 3"):
            print("   ", str(r)[:300])
