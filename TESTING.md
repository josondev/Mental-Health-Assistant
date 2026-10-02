# Testing Guide — Mental Health RAG Assistant

This document explains every test layer in the project, what each one covers,
how to run them locally, and what to check before deploying to Streamlit Cloud.

---

## Test layers at a glance

| Layer | File | Needs API keys? | Needs DB? | Speed |
|---|---|---|---|---|
| DB unit tests | `tests/test_database.py` | No | No (SQLite in-memory) | ~13 s |
| App–DB integration | `tests/test_app_db_integration.py` | No | No (SQLite in-memory) | < 1 s |
| Input guard unit tests | `tests/test_input_guard.py` | No | No | < 1 s |
| RAG evaluation pipeline | `evaluation/` | **Yes** | No | ~60 s per run |
| Live smoke test | `evaluation/smoke_test.py` | **Yes** | No | ~6 s |

---

## Prerequisites

Make sure you are in the project root and your environment has all dependencies:

```powershell
pip install -r requirements.txt
```

You do **not** need any API keys to run the pytest suite.

---

## 1. Run the full pytest suite (recommended first step)

```powershell
python -m pytest -v
```

Expected output:

```
69 passed in ~13s
```

To run a single test file:

```powershell
python -m pytest tests/test_database.py -v
python -m pytest tests/test_app_db_integration.py -v
python -m pytest tests/test_input_guard.py -v
```

To run a single test by name:

```powershell
python -m pytest tests/test_database.py::TestCascadeDelete -v
```

---

## 2. What each test file covers

### `tests/test_database.py` — 31 tests

Tests every public function in `src/database.py` against a fresh
SQLite in-memory database. No network calls, no real Postgres needed.

| Class | What it checks |
|---|---|
| `TestInitDb` | Happy path, missing URL, bad URL, idempotent double-call |
| `TestGetOrCreateSession` | Create, idempotent second call, `updated_at` touched, unavailable DB |
| `TestSaveMessage` | User turn, assistant turn with audit fields, optional fields omitted |
| `TestLoadSessionMessages` | Ordering, role preservation, doc ID round-trip, latency, session isolation, missing DB |
| `TestDeleteSessionMessages` | Wipes target only, safe on empty session, missing DB |
| `TestIsDbAvailable` | Reflects engine state before/after init, after failed init |
| `TestCascadeDelete` | Deleting a session row removes its messages via `ON DELETE CASCADE` |

The SQLite engine has `PRAGMA foreign_keys=ON` enabled so cascade behaviour
matches Postgres exactly.

### `tests/test_app_db_integration.py` — 16 tests

Tests the DB wiring inside `src/app.py` without touching any external API.
All LangChain, Pinecone, and Gemini imports are replaced with in-process stubs
at import time. The RAG chain is replaced with a stub that returns a fixed
answer and a fake retrieved document.

| Class | What it checks |
|---|---|
| `TestAskQuestionForEvaluationDb` | Writes 2 rows per call, correct role/content, doc IDs persisted, latency persisted, no write when `session_id=None`, result dict has all keys, DB failure swallowed |
| `TestAskQuestionStreamedDb` | Same checks for the streaming path, assembled response correct |

### `tests/test_input_guard.py` — 22 tests

Tests `src/input_guard.py` — the pure validation function that runs before
any input reaches the LLM or database.

| Class | What it checks |
|---|---|
| `TestValidInput` | Normal messages, stripping, exact boundary (`MAX_INPUT_CHARS` chars accepted) |
| `TestEmptyInput` | Blank, whitespace-only, newlines-only, tabs-only all raise `EmptyInputError` |
| `TestOverLengthInput` | One over limit raises, far over limit raises, error message contains char count and limit, whitespace padding cannot bypass the guard |
| `TestConstant` | `MAX_INPUT_CHARS` is a positive, reasonable value |

---

## 3. Live smoke test (requires API keys)

The smoke test runs one real question through the full RAG chain and prints
the answer and retrieved documents. Use this to verify the system works
end-to-end after a deployment or dependency change.

```powershell
# From the project root:
python evaluation/smoke_test.py
```

**Requires** `PINECONE_API_KEY`, `GOOGLE_API_KEY`, and `PINECONE_REGION` set
in `src/.env` or in your shell environment.

Expected output:

```
QUESTION
What are some grounding techniques that may help with anxiety?

ANSWER
<response from Gemini>

RETRIEVED DOCUMENTS
Count: 5
...
```

If you see an error about the Pinecone index not being found, the index has
not been populated yet — see the data preparation section in `README.md`.

---

## 4. RAG evaluation pipeline (requires API keys)

The evaluation pipeline measures retrieval quality, faithfulness, empathy,
crisis safety, and latency across a labelled test set.

### Step 1 — Generate predictions

```powershell
python evaluation/generate_predictions.py
```

Runs all 10 prompts in `evaluation/mental_health_tests.jsonl` through the
live RAG chain and writes results to `evaluation/rag_predictions.jsonl`.

### Step 2 — Label predictions manually

```powershell
python evaluation/manual_review.py
```

Walks through each unreviewed prediction interactively. Labels are saved
after every record so you can stop and resume safely with `Ctrl+C`.

Fields you will be asked for each prediction:

- Which retrieved documents were actually relevant (by number)
- Faithfulness score (0 / 0.5 / 1)
- Answer relevance score (0 / 0.5 / 1)
- Empathy score (1–5)
- Whether the answer treated it as an immediate crisis (y/n)
- Whether the answer contained harmful advice (y/n)
- Whether professional boundaries were maintained (y/n)
- For out-of-domain prompts: whether the refusal was correct (y/n)

### Step 3 — Audit label coverage

```powershell
python evaluation/audit_rag_labels.py --predictions evaluation/rag_predictions.jsonl
```

Shows how many records have labels for each metric field. Run this before
calculating metrics to confirm all records are reviewed.

### Step 4 — Calculate metrics

```powershell
python evaluation/rag_eval.py `
  --predictions evaluation/rag_predictions.jsonl `
  --output evaluation/rag_eval_results.json
```

Writes aggregate metrics to `evaluation/rag_eval_results.json`.

Only report a metric if its `metric_counts` entry matches the number of
reviewed examples. A score based on 1 out of 10 records is not reportable.

---

## 5. Testing with a real Postgres database locally

To test the DB layer against a real Postgres instance (e.g. a local Docker
container or your Supabase project) instead of SQLite:

```powershell
$env:DATABASE_URL = "postgresql://user:password@localhost:5432/mental_health_test"
python -m pytest tests/test_database.py -v
```

The test suite will use your `DATABASE_URL` if set. The in-memory SQLite
fallback only activates when the env var is absent.

> **Note:** Running against a real database creates and drops schema using
> `CREATE TABLE IF NOT EXISTS`. It does not drop tables after the run.
> Use a dedicated test database, not your production one.

---

## 6. Checking Streamlit Cloud deployment

There is no automated test for the Streamlit UI layer. After deploying, verify
these manually:

| Check | How |
|---|---|
| App loads without errors | Open the deployed URL |
| Sidebar shows `💾 History: saved to database` | Confirm `DATABASE_URL` is set in Streamlit Cloud secrets |
| Sidebar shows `⚠️ History: local only` if DB not configured | Remove `DATABASE_URL` from secrets temporarily |
| History survives a page refresh | Send a message, refresh, confirm it reappears |
| Clear button wipes history | Click it, refresh, confirm history is gone |
| Over-long input is rejected | Paste 2001+ characters and submit — warning should appear |
| Out-of-domain question is refused | Ask "What is the capital of France?" — bot should redirect |

---

## 7. File structure summary

```
Mental-Health-Assistant/
├── src/
│   ├── app.py              # RAG chain + DB wiring
│   ├── database.py         # SQLAlchemy DB layer
│   └── input_guard.py      # Input validation (pure, no Streamlit)
│
├── tests/
│   ├── __init__.py
│   ├── test_database.py         # 31 DB unit tests
│   ├── test_app_db_integration.py  # 16 integration tests
│   └── test_input_guard.py      # 22 input guard tests
│
├── evaluation/
│   ├── mental_health_tests.jsonl   # 10 labelled test prompts
│   ├── generate_predictions.py     # Run RAG on test set
│   ├── manual_review.py            # Interactive labelling
│   ├── audit_rag_labels.py         # Coverage check
│   ├── rag_eval.py                 # Metric calculation
│   ├── rag_predictions.jsonl       # Generated (gitignored)
│   └── rag_eval_results.json       # Results (gitignored)
│
├── pytest.ini          # testpaths=tests, pythonpath=src
└── requirements.txt    # includes pytest==8.3.4
```
