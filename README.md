# RAG over 10-K risk factors

A small retrieval-augmented question-answering pipeline for an equity research analyst. You ask a question about the risk factors of 8 large US companies, and the pipeline returns an answer with **cited sources (file + chunk id)**, a **confidence score**, and a **refusal** when nothing relevant is found.

Built as the capstone project (Brief A) of the *Applied NLP for Business* master class at ESMT Berlin, 2026.

## How it works

```
question ─┬─ BM25 (keywords) ──┐
          └─ dense (Chroma) ───┴─ Reciprocal Rank Fusion ─ cross-encoder rerank
                                     ─ guardrail: drop chunks scoring below −2 (none left → refuse)
                                     ─ local LLM answer with [n] citations + confidence
```

| Step | Choice |
|---|---|
| Chunking | `RecursiveCharacterTextSplitter`, 600 characters, 80 overlap → 32 chunks with `file::chunk` ids |
| Dense retrieval | `all-MiniLM-L6-v2` embeddings in an in-memory Chroma collection |
| Keyword retrieval | `rank_bm25` (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Guardrail | rerank score threshold −2; no chunk above it → refusal without calling the LLM |
| Confidence | sigmoid of the best rerank score |
| LLM | `llama3.2:3b-instruct` via Ollama (local), chunks labelled with source + company |

## Results (6 test questions)

| # | Tests | Answer | Citation correct? | Hallucination? |
|---|---|---|---|---|
| Q1 | Precise fact | ✅ Correct | Yes | No |
| Q2 | Company + exact terms | ✅ Correct | Yes | No |
| Q3 | Two documents | ✅ Correct | Yes | No |
| Q4 | Paraphrase | ❌ Wrongly refused | n/a | No |
| Q5 | Hallucination trap | ✅ Refused | n/a | No |
| Q6 | Off-topic | ✅ Refused | n/a | No |

The full evaluation and failure analysis are in the notebook. The design choices, limitation and risks are in [writeup.md](writeup.md).

## Setup

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download).

```bash
# 1. Python environment (creates .venv from pyproject.toml + uv.lock)
uv sync

# 2. Local LLM
ollama pull llama3.2:3b-instruct-q4_K_M
```

No Ollama? Copy `.env.example` to `.env` and set a Groq key or any OpenAI-compatible endpoint. `course_utils.get_llm_client()` picks the backend automatically.

On the first run, the embedding model and the cross-encoder (~170 MB) are downloaded from Hugging Face.

## Run

```bash
uv run jupyter lab capstone.ipynb
```

or headless, top to bottom:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace capstone.ipynb
```

A full run takes about 1–2 minutes on an Apple Silicon laptop.

## Repository layout

| Path | Contents |
|---|---|
| `capstone.ipynb` | The pipeline and the evaluation |
| `writeup.md` | One-page write-up: problem, design choices, limitation, business risk |
| `course_utils.py` | LLM-backend helper from the course (Ollama / Groq / local server) |
| `data/sample_10k_excerpts/` | The 8 10-K risk-factor excerpts |
| `data/LICENSES.md` | Data source and license |

## Data and attribution

- The 10-K excerpts are **short paraphrases written for teaching** by the course authors, based on public SEC filings (public domain in the US). They are not verbatim filings and shouldn't be cited as financial disclosures. See [data/LICENSES.md](data/LICENSES.md).
- `course_utils.py` and the data come from the *Applied NLP for Business* course materials (code: MIT, materials: CC BY-SA 4.0).

## License

Code: MIT. Notebook text, write-up and data: CC BY-SA 4.0. See [LICENSE](LICENSE).
