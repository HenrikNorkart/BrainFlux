# T3 — Making clinical-literature retrieval live on CAIR

HQ #126. Session of 2026-09-17 (04:08–04:25 UTC+02:00), unattended nightshift.

**Status: partially complete.** The knowledge channel is now provably live
(AC1, AC2, AC4, AC5 met). The web-search channel is **not** live and cannot be
made live from here — the Serper account is out of credits (AC3 blocked, needs
Henrik).

---

## 1. Root cause — three bugs, not one

The item described one hardcode. There were three, and together they explain the
whole dead-retrieval run.

### 1a. Hardcoded Chroma persist path

`rogueone/llm/agents/knowledge.py` ignored `cfg.knowledge_db_path` entirely:

```python
persist_directory=str("/workspaces/BrainFlux/chroma_db_medical_knowledge")
```

That path only exists inside Henrik's devcontainer. Anywhere else Chroma
silently auto-creates an empty store.

### 1b. Hardcoded collection name

`rogueone/llm/agents/scientist.py` also ignored the config:

```python
self.knowledge_agent = KnowledgeAgent(cfg=self.cfg_experiment,
                                      collection_name="cardiac_arrest")
```

So the effective store was `chroma_db_medical_knowledge` + collection
`cardiac_arrest`. **That combination is empty on CAIR, in both checkouts.**
Verified directly against the Chroma sqlite files (read-only, in a container):

| store | collection | embeddings |
|---|---|---|
| `/raid/home/henrbr16/BrainFlux_run/chroma_db_medical_knowledge` | `cardiac_arrest` | **0** |
| `/raid/home/henrbr16/code/BrainFlux/chroma_db_medical_knowledge` | `cardiac_arrest` | **0** |
| `/raid/home/henrbr16/BrainFlux_run/chroma_db` | `medical_knowledge` | **120** |
| `/raid/home/henrbr16/BrainFlux_run/chroma_db` | `tester_knowledge_collection` | 83 |
| `/raid/home/henrbr16/BrainFlux_run/chroma_db` | `feature_extraction_literature` | 40 |

The real clinical corpus was sitting one directory up the whole time, under a
different collection name. Sample titles: *2023 AHA Focused Update on Adult
ACLS*, *ERC/ESICM Guidelines 2021: Post-resuscitation care*, *Neurological
prognostication after cardiac arrest*, *Prognostication in Acute Neurological
Emergencies*. It is exactly the right corpus for this paper.

Note the 24 h config also pointed `knowledge_db_path` at
`/workspace/chroma_db/cardiac_arrest_knowledge_db`, a directory that does not
exist either — so fixing only bug 1a would still have produced an empty store.

### 1c. Serper key overwritten at import time

`rogueone/llm/agents/web_search.py` line 9 did:

```python
os.environ["SERPER_API_KEY"] = r"beb4ce298263eb3a1f442507b051d9599e028fca"
```

This runs at **import** time, so it clobbers whatever `.secret.env` supplies.
The key Henrik added after the 24 h run therefore had no effect at all. (That
literal is also a live-looking secret committed to the repo; it should be
rotated regardless of what happens with the credits.)

---

## 2. Fixes

Branch `fix/threshold-train-optimization-v2`:

- **`db9e184`** — path from `cfg.knowledge_db_path`, collection from
  `cfg.knowledge_db_collection_name`, Serper key from the environment only, and
  a new `rogueone/utils/retrieval_guards.py` that raises `DeadRetrievalChannel`
  at agent construction when the store is empty or the key is missing. Adds
  `tmp/t3_retrieval_smoke.py`.
- **`4636f6b`** — the guard now *probes* Serper with a real one-credit search
  instead of only checking that a key exists (see §4 for why that matters).

Escape hatch: `ROGUE_ONE_ALLOW_DEAD_RETRIEVAL=1` downgrades both guards to
warnings, for a deliberate no-retrieval ablation. It must be recorded in the run
config when used.

Config change on CAIR, `phantom_menace/config_24h.yml`:

```
knowledge_db_collection_name: cardiac_arrest                        -> medical_knowledge
knowledge_db_path:            /workspace/chroma_db/cardiac_arrest_knowledge_db
                                                                    -> /workspace/chroma_db
```

**Commit synced onto CAIR: `4636f6b7c5f8022055d1e1da9637248703f6ba09`**, into
`/raid/home/henrbr16/BrainFlux_run`. See §5 for how, and why not a full checkout.

---

## 3. Proof the knowledge channel is live (AC2)

Run: `~/t3_launch.sh` → container `brainflux_t3` from `brainflux_pm:latest`,
GPUs 0 and 1 only (4–7 were held by planbench), vLLM `openai/gpt-oss-20b` on
:8000 and `Qwen/Qwen3-Embedding-4B` on :8001. Log: `~/t3_retrieval_live.log`.
Script: `tmp/t3_retrieval_smoke.py`.

```
[2026-09-17T04:17:01+00:00] ===== PROOF 1/4 -- knowledge store embedding count =====
[retrieval-guard] knowledge store 'medical_knowledge' at /workspace/chroma_db: 120 embeddings
embedding count            : 120

[2026-09-17T04:17:02+00:00] ===== PROOF 2/4 -- similarity search returns a real passage =====
query                      : What vital-sign patterns in the first 24 hours predict outcome after cardiac arrest?
documents returned         : 3
```

Passage 1 (1313 chars), *Prognostication in Acute Neurological Emergencies*:

> This study demonstrates that acute neurological prognostication within the
> first 24 hours is highly variable and often inaccurate among neurologists,
> regardless of their training level or subspecialty. […] the collective
> prediction of the group (the 'wisdom of the crowd') was more accurate,
> matching the outcome in 60% of cases.

Passage 3 (1081 chars), *ERC/ESICM Guidelines 2021: Post-resuscitation care*:

> **Seizure Control:** Seizures occur in 20-30% of cardiac arrest patients and
> usually indicate severe brain injury. Continuous EEG monitoring is useful for
> diagnosis. […] **Temperature Control:** Targeted Temperature Management (TTM)
> is a key intervention. A constant target temperature between 32°C and 36°C
> for at least 24 hours is recommended for unresponsive adults after OHCA or
> IHCA.

This is a real retrieval of real, on-topic clinical text through the same
embedding path the agents use. AC2 met.

---

## 4. The web-search channel is dead — out of credits (AC3 NOT met)

With the import-time clobber removed, the key from `.secret.env` (40 chars,
ending `ac4f`) reaches Serper. It is accepted as a key and then rejected for
billing. Both POST and GET, same answer:

```
POST https://google.serper.dev/search  -> 400
{"message":"Not enough credits","statusCode":400}
GET  https://google.serper.dev/search  -> 400
{"message":"Not enough credits","statusCode":400}
```

**This is a Henrik decision, not something I should act on:** it needs credits
bought on the Serper account, or a different search provider wired in. I have
not touched billing or swapped providers.

Why it matters beyond T3: `_run_search_agent` catches the failure and the agent
reports "no information found". A web-search channel that is out of credits is
therefore **indistinguishable from a web-search channel that found nothing** —
the same invisibility that hid the dead knowledge store. Commit `4636f6b`
closes that hole: the launch probe now fails loudly on any non-200 from Serper,
credits included.

Consequence for the discovery run: it can go ahead with the knowledge channel
live and web search still dead, but that is a **partial** un-handicapping. The
resulting run would be a stronger lower bound than the 24 h run, not a fair
test. My recommendation is to buy credits before launching, but that is
Henrik's call and T5 should not be started on the assumption it is fixed.

---

## 5. What the sync would have broken (AC6)

`/raid/home/henrbr16/BrainFlux_run` was on `fix/threshold-train-optimization`
at `3abcc24` with **eleven tracked files carrying uncommitted local edits** that
exist nowhere in git:

```
Dockerfile, phantom_menace_main.py, phantom_menace/agents/tester.py,
rogueone/llm/agents/{extractor,knowledge,scientist,summary,tester,web_search}.py,
scripts/run_vllm_{oss120,qwen3_embedd}.sh
```

These are not cosmetic. They include the recursive special-token stripping in
every `_strip_reasoning_items` filter (`<|end|>`, `<|channel|>`, … must be
scrubbed out of *all* string fields or vLLM 400s), the empty-content-item filter,
and `session=` disabled in `scientist.py` with the note *"SQLiteSession causes
poisoned-message 500 errors"*. **A `git checkout` / `reset --hard` of the branch
would have deleted the fixes that make the loop run at all on gpt-oss.**

So I did not sync the branch. I fetched only the two genuinely new files
(`rogueone/utils/retrieval_guards.py`, `tmp/t3_retrieval_smoke.py`) and applied
the three targeted edits on top of the existing working tree with an
anchor-checked patch script that aborts if any anchor is missing. Originals are
backed up in `~/t3_backup/` (`knowledge.py`, `scientist.py`, `web_search.py`,
`config_24h.yml`).

**Follow-up filed:** those eleven files of CAIR-only work need committing to
git. Until then the box and the repo disagree about what the pipeline is, and
nobody can reproduce a run from the repo alone.

---

## 6. Acceptance criteria

| # | criterion | status |
|---|---|---|
| 1 | `knowledge_db_path` respected | **met** — path and collection both come from config now |
| 2 | embedding count > 0 **and** a real passage retrieved | **met** — 120 embeddings, 3 on-topic passages, §3 |
| 3 | one successful `search_tool()` call captured | **NOT met** — Serper out of credits, §4 |
| 4 | startup guards in place, dead channel fails loudly | **met** — `retrieval_guards.py`, both channels, `db9e184` + `4636f6b` |
| 5 | this file, with proofs, timestamps and synced commit | **met** |
| 6 | stop and report if the sync breaks something | **met** — it would have, §5; did not sync, reported instead |

---

## 7. Housekeeping

- GPUs used: 0 and 1, released. `docker ps` after the run shows only
  `planbench-*`; GPUs 0–3 report 0 MiB.
- `~/brainflux_nightshift.lock` taken at session start, released at end.
- Nothing under `paper/` was touched. Nothing was pushed to Overleaf.
