# Evals — Golden Dataset Evaluation Framework

Run the AI Finance Assistant against a curated golden dataset and record
results as a LangSmith Experiment.

---

## Prerequisites

Set the following environment variables (copy `.env.example` → `.env` and fill
in the blanks):

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key used by the application and the LLM judge |
| `LANGSMITH_API_KEY` | API key from [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys |
| `LANGSMITH_PROJECT` | LangSmith project name (e.g. `ai-finance-assistant`) |
| `LANGSMITH_TRACING` | Must be `true` to enable trace upload |

The runner raises a clear `EnvironmentError` listing any missing variables
before it attempts any network calls.

---

## How to Run

```bash
uv run python evals/runner.py
```

Optional: override the experiment name prefix:

```bash
uv run python evals/runner.py --experiment-prefix my-experiment
```

Each run creates a new timestamped experiment (e.g.
`finance-assistant-20240115T143022`) under the
`ai-finance-assistant-golden` dataset in LangSmith.  Examples are uploaded
idempotently — re-running does not create duplicates.

---

## Viewing Results in LangSmith

1. Open [smith.langchain.com](https://smith.langchain.com).
2. Navigate to **Datasets** in the left sidebar.
3. Click `ai-finance-assistant-golden`.
4. Open the **Experiments** tab to see all runs.
5. Click any experiment to see per-example scores and evaluator comments.
6. Use the **Compare** button to diff two experiments side-by-side.

---

## Adding New Dataset Entries

Open `evals/evaluation_dataset.json` and append a new JSON object to the
array.  Every entry must include:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Unique stable identifier (e.g. `finance_0002`). Used as the LangSmith Example ID seed. |
| `question` | string | Yes | The user question sent to the assistant. |
| `expected_behavior` | string | Yes | `"answer"` — the assistant should give a substantive reply. `"refuse"` — the assistant should emit the out-of-scope refusal. |
| `expected_facts` | array of strings | Yes | Facts that must appear in the response (even if paraphrased). Empty list `[]` is valid. |
| `forbidden_facts` | array of strings | No | Facts that must NOT appear in the response. Empty list `[]` is valid. |
| `expected_source_ids` | array of strings | No | URLs that must appear (as substrings) in the response. Leave empty `[]` if you do not want citation checking. |
| `expected_routes` | array of strings | No | Which route(s) the router should select: `"financial_concepts"`, `"realtime_quotes"`, `"out_of_scope"`. Informational only — not evaluated by the runner. |
| `expected_nodes` | array of strings | No | Which LangGraph nodes should execute (`"router_node"`, `"financial_concepts_node"`, `"realtime_quotes_node"`, `"synchronizer_node"`). Informational only. |
| `evaluation_scopes` | array of strings | No | Which evaluation layers apply: `"application"`, `"retrieval"`, `"generation"`. Informational only. |
| `category` | string | No | Semantic category for filtering (e.g. `"definition"`, `"comparison"`). |
| `tags` | array of strings | No | Free-form tags for grouping (e.g. `["stocks", "beginner"]`). |
| `difficulty` | string | No | `"easy"`, `"medium"`, or `"hard"`. |
| `risk_level` | string | No | `"low"`, `"medium"`, or `"high"` — flags safety-sensitive questions. |

**Minimal valid entry:**

```json
{
  "id": "finance_0002",
  "question": "What is a bond?",
  "expected_behavior": "answer",
  "expected_facts": ["loan to a company or government", "fixed interest"],
  "forbidden_facts": [],
  "expected_source_ids": []
}
```

---

## Metrics Reference

### `answer_correctness`

**What it measures:** Whether the response contains all expected facts and
avoids all forbidden facts, as judged by `gpt-4o-mini`.

| Score | Meaning |
|---|---|
| `1.0` | All `expected_facts` found (even if paraphrased) AND no `forbidden_facts` present. |
| `0.0` | At least one expected fact missing OR at least one forbidden fact present. |
| `1.0` (skipped) | `expected_behavior != "answer"` — metric does not apply to refusal entries. |

The evaluator comment includes per-fact FOUND/MISSING status and the judge's
reasoning.

---

### `refusal_correctness`

**What it measures:** Whether the assistant refused or answered as expected.
Deterministic — no LLM call.

The refusal is detected by the presence of the phrase
`"does not have the ability"` in the response, which is the fixed string
emitted by `synchronizer_node` for out-of-scope requests.

| Score | Meaning |
|---|---|
| `1.0` | `expected_behavior == "answer"` and response does NOT contain the refusal phrase. |
| `1.0` | `expected_behavior == "refuse"` and response DOES contain the refusal phrase. |
| `0.0` | Mismatch between expected and actual refusal behavior. |

---

### `citation_presence`

**What it measures:** Whether every URL in `expected_source_ids` appears
somewhere in the response (normalized substring match).

Normalization: both the expected URL and the response text are lowercased and
trailing slashes are stripped before comparison, so minor formatting
differences do not cause false negatives.

| Score | Meaning |
|---|---|
| `1.0` | All expected URLs found in the response. |
| `0.0` | At least one expected URL is missing. |
| `1.0` (skipped) | `expected_behavior != "answer"`, or `expected_source_ids` is empty/absent. |

The evaluator comment lists found URLs with up to 100 characters of
surrounding context and explicitly names any missing URLs.
