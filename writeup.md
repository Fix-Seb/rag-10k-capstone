# Capstone: RAG over the 10-Ks (Brief A)

## Problem & user

- **User:** an equity research analyst covering several large caps.
- **Decision:** which risks to flag in a company note or a peer comparison, e.g. "Who is exposed to Russia sanctions?"
- **Today:** reading every 10-K's risk factors by hand and searching with Ctrl+F, which misses anything worded differently.
- **After:** the analyst asks in plain English and gets an answer in seconds. Every statement is cited with file + chunk id, so it can be checked in one click. If nothing relevant is found, the tool refuses instead of guessing.
- **ROI signal:** time per cross-company risk question (minutes → seconds) and the share of answers that need no correction after checking the citation.

## Design choices

- **Chunking:** 600 characters with 80 overlap → 32 chunks. The excerpts are short (~250 words), so bigger chunks would mean one chunk per document. Each chunk keeps its file, company and an id like `Apple_10k_excerpt.txt::2`.
- **Hybrid retrieval:** BM25 finds exact terms, dense embeddings (`all-MiniLM-L6-v2` in Chroma) find similar meaning. Reciprocal Rank Fusion combines both rankings without needing comparable scores.
- **Reranking:** a cross-encoder reads question and chunk together and gives a more precise relevance score.
- **Guardrail:** chunks scoring below −2 are dropped. If none are left, the tool refuses and the LLM is never called. I started at 0, but that wrongly refused the Tesla minerals question (correct chunk scored −1.37).
- **Confidence:** sigmoid of the best rerank score (0–1).
- **Prompt:** each chunk is labelled with its source and company, and the LLM must cite `[n]`. Without labels it attributed Tesla's "vehicle production" risk to Apple. A "write full sentences" rule fixed answers that only repeated the labels.
- **LLM:** local `llama3.2:3b` via Ollama: free, and no data leaves the machine.

## Evaluation

Six questions, each testing something different: a precise fact, a company filter, two documents, a paraphrase, a hallucination trap ("Who is Tesla's CEO?", which the excerpts never answer) and an off-topic question.

- **Result:** 5 of 6 correct, all citations correct, **no hallucinations**. Even with the guardrail switched off, the LLM answered "I don't know" to both trap questions, so the prompt works as a second line of defence.
- **Failure:** the paraphrase question ("tighter household budgets") was wrongly refused. The reranker didn't connect it to "consumer spending" and scored the right Amazon/Walmart chunks far below the threshold.

## One limitation

- **Wrong refusals caused by a fixed threshold and a small, general-purpose reranker.** The threshold is a trade-off: 0 refused a correct answer, while −100 lets everything through. −2 is tuned on only six questions. The reranker was trained on web search (MS MARCO), not on financial text, so it misses paraphrases.
- **Next step:** build a labelled set of 30–50 analyst questions, choose the threshold based on precision and recall, and compare a larger or finance-tuned reranker on the same set.

## One business risk & mitigation

- **Hallucination / misattribution.** A risk assigned to the wrong company could end up in a client note and damage trust. We saw it happen once (Tesla → Apple).
- **Mitigations:**
  1. The relevance threshold with refusal.
  2. Source and company labels plus a strict prompt rule against mixing companies.
  3. Mandatory citations, so every claim can be checked.
  4. Analyst review before anything is published.
  5. A clear note that confidence measures retrieval relevance, not whether the answer is correct.

A wrong refusal only costs the analyst some time. A hallucination costs trust, so I deliberately err on the strict side.