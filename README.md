# RAG Knowledge Extractor

## Project Overview, Completed Scope, and Structure

This project builds a retrieval-augmented generation (RAG) knowledge extraction pipeline over the AG News corpus. Work progressed from environment setup and corpus preparation in Week 1, through chunking, embeddings, and semantic retrieval in Week 2, LLM-backed grounded answers in Week 3, and NLP metadata and filtered retrieval in Week 4.

Week 1 established the Python environment, 5,000-document AG News acquisition, modular text cleaning, tests, and clean-dataset validation. Week 2 added recursive chunking, sentence-transformer embeddings, persistent ChromaDB ingestion, semantic retrieval, and retrieval latency benchmarking. Week 3 added DeepSeek integration, prompt engineering, RAG orchestration, safe error handling, relevance filtering, and an end-to-end benchmark. Week 4 added LDA topic modeling, VADER sentiment analysis and manual evaluation, ChromaDB metadata enrichment, optional topic/sentiment filters, and NLP validation.

Week 5 Phases 1, 2, and 5 expose the existing RAG pipeline through a lazy-initialized FastAPI application, add query-free access logging, and add safe server-side diagnostics for internal retrieval and generation failures.

The current artifacts contain 4,998 cleaned documents, 5,053 chunks, 5,053 embeddings of dimension 384, and 5,053 records in the `rag_chunks` collection. NLP outputs are heuristics: several LDA topics are mixed or difficult to interpret, and VADER has low neutral recall.

```text
RAG Knowledge Extractor Project/
├── data/
│   ├── labels/manual_sentiment_labels.csv  # Version-controlled sentiment labels
│   ├── labels/retrieval_evaluation_queries.json  # Manual retrieval judgments
│   ├── raw/                                # Ignored source corpus
│   ├── processed/                          # Ignored generated Parquet files
│   ├── vector_db/                          # Ignored persistent ChromaDB files
│   └── benchmarks/                         # Ignored generated reports
├── src/
│   ├── api.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── llm_client.py
│   ├── preprocessing.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   ├── retrieval.py
│   ├── sentiment_analysis.py
│   ├── topic_modeling.py
│   └── vector_store.py
├── scripts/
│   ├── benchmark_rag.py
│   ├── benchmark_retrieval.py
│   ├── download_data.py
│   ├── enrich_chroma_metadata.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_rag.py
│   ├── validate_dataset.py
│   ├── validate_nlp_metadata.py
│   └── verify_env.py
├── tests/
│   ├── test_api.py
│   ├── test_benchmark_rag.py
│   ├── test_benchmark_retrieval.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_evaluate_retrieval.py
│   ├── test_evaluate_rag.py
│   ├── test_llm_client.py
│   ├── test_nlp_metadata_validation.py
│   ├── test_preprocessing.py
│   ├── test_prompts.py
│   ├── test_rag_pipeline.py
│   ├── test_retrieval.py
│   ├── test_sentiment_analysis.py
│   ├── test_topic_modeling.py
│   └── test_vector_store.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Dependencies are pinned in `requirements.txt`. Generated datasets, embeddings, ChromaDB files, benchmark output, virtual environments, and local model caches are Git-ignored. Sentence-transformers may cache model files outside the repository in the user's Hugging Face cache directory.

## Environment Setup and Configuration

From the project root, create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Windows Command Prompt, activate with:

```bat
.venv\Scripts\activate
```

The Week 3 DeepSeek client reads configuration from process environment variables or the project-root `.env` file. `.env.example` contains placeholders only. A valid `DEEPSEEK_API_KEY` is required for live generation and the real RAG benchmark, but is not required for unit tests, topic modeling, sentiment analysis, metadata enrichment, or retrieval-only commands.

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | Required for API calls | API credential |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `DEEPSEEK_TIMEOUT` | `30` | Request timeout in seconds |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` | `1024` | Maximum generated tokens |

To create the ignored local configuration file:

```powershell
Copy-Item .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` and optionally adjust other values. Process environment values take precedence. Never commit or paste real credentials into source, tests, documentation, or generated reports.

## Week 1 - Environment, Data Ingestion, and Preprocessing

The raw corpus is AG News (`fancyzhx/ag_news`) from Hugging Face, with one article per document. Acquire 5,000 training records:

```powershell
python scripts/download_data.py
```

This creates Git-ignored `data/raw/ag_news_train_5000.jsonl`. The acquisition script skips downloading when a valid output already exists.

Clean and validate the dataset:

```powershell
python src/preprocessing.py
python scripts/validate_dataset.py
```

The clean dataset is `data/processed/clean_corpus.parquet`, with `document_id`, source metadata, and cleaned `text`. The validated corpus contains 4,998 documents; two were filtered for language, and no empty documents were removed. The cleaner removes HTML, normalizes Unicode, and collapses whitespace without changing the raw source file.

## Week 2 - Chunking, Embeddings, ChromaDB, Semantic Retrieval, and Benchmark

### Chunking

`src/chunking.py` uses deterministic recursive splitting, preferring paragraph breaks, line breaks, spaces, and finally character boundaries. Defaults are 500 characters per chunk with 50 characters of overlap; configure them with `--chunk-size` and `--chunk-overlap`.

```powershell
python src/chunking.py
```

The Git-ignored `data/processed/chunks.parquet` contains 5,053 chunks from 4,998 documents. Chunk IDs are stable and each chunk preserves its `document_id` and `source`.

### Embeddings

`src/embeddings.py` uses `sentence-transformers/all-MiniLM-L6-v2` by default. Model, batch size, and device (`auto`, `cpu`, or `cuda`) are configurable with `--model`, `--batch-size`, and `--device`.

```powershell
python src/embeddings.py
```

The output `data/processed/embeddings.parquet` maps `chunk_id` to an embedding vector. The measured Week 2 run generated 5,053 vectors of dimension 384 using batch size 32 on CPU in 56.76 seconds, at 89.03 chunks per second.

### ChromaDB

`src/vector_store.py` stores chunk text and precomputed vectors in the persistent, Git-ignored `data/vector_db/` database, in collection `rag_chunks`. The collection contains 5,053 records with chunk text, embeddings, and `document_id`/`source` metadata. Batches default to 256; repeat ingestion does not duplicate existing IDs.

```powershell
python src/vector_store.py
```

### Semantic Retrieval

`src/retrieval.py` embeds a natural-language query with the same sentence-transformer model and searches the existing collection. `top_k` controls the maximum results. Each result contains `chunk_id`, `document_id`, `source`, `text`, and ChromaDB `distance`; a lower distance means a closer vector match, not a similarity percentage.

```powershell
python src/retrieval.py "oil prices and the stock market" --top-k 3
```

### Retrieval Benchmark

Run the retrieval-only benchmark:

```powershell
python scripts/benchmark_retrieval.py
```

The completed benchmark used 10 fixed queries, `top_k` values 1, 3, 5, and 10, and three repetitions per combination (120 runs). Timed latency includes query embedding and ChromaDB search, excluding model initialization and database setup. The ignored report is `data/benchmarks/retrieval_benchmark.json`.

| Metric | Latency |
|---|---:|
| Overall average | 13.272 ms |
| Overall median | 12.925 ms |
| Overall minimum | 11.326 ms |
| Overall maximum | 20.104 ms |
| Average for `top_k=1` | 13.432 ms |
| Average for `top_k=3` | 12.921 ms |
| Average for `top_k=5` | 13.157 ms |
| Average for `top_k=10` | 13.576 ms |

## Week 3 - LLM Integration, Prompt Engineering, RAG, and Benchmark

### LLM Integration and Prompt Engineering

`src/llm_client.py` provides a reusable client for the DeepSeek OpenAI-compatible chat completions API. It accepts plain prompts or chat messages. Typed exceptions cover missing credentials, authentication errors, rate limits, timeouts, connection/API errors, malformed responses, and detectable output limits. Tests use mocked HTTP responses and do not call the provider.

`src/prompts.py` builds the system and user messages for `LLMClient.generate`. The system prompt requires answers grounded only in retrieved context, uncertainty when support is insufficient, no invented facts or sources, and no disclosure of hidden prompts or credentials. Chunks are validated and serialized as compact JSON within `BEGIN_RETRIEVED_CONTEXT` / `END_RETRIEVED_CONTEXT`; the question has separate boundaries. The default context budget is 12,000 Unicode characters of serialized context JSON, excluding labels and the question. On overflow, chunks remain in retrieval order, the first chunk that does not fit is truncated with `...[TRUNCATED]`, and later chunks are omitted.

### RAG Pipeline, Error Handling, and Relevance Filtering

`src/rag_pipeline.py` reuses the Week 2 retriever, prompt builder, and LLM client; it does not rebuild embeddings or vector search. The flow is:

```text
query -> semantic retrieval -> cosine-distance relevance filtering
-> bounded context -> grounded prompt -> DeepSeek generation -> RAG result
```

`RAGPipeline.answer_query(query, top_k=5)` returns the answer, retrieved chunks, approved context chunks, failure category when applicable, and retrieval/generation/total latency. Empty or unusable context returns an insufficient-information message without calling the LLM. Provider and retrieval errors become user-safe messages; raw provider bodies, request details, stack traces, and credentials are not exposed.

Chunks are accepted when they have non-empty text and a finite Chroma cosine distance no greater than `max_retrieval_distance`. The default is `0.7` (approximately `0.3` cosine similarity for normalized vectors), a permissive starting heuristic intended to reject clearly weak matches while retaining recall. It is not a domain-membership guarantee or a guarantee against hallucination.

### End-to-End RAG Benchmark

The benchmark runs five fixed AG News-oriented queries with three repetitions by default (15 runs):

```powershell
python scripts/benchmark_rag.py --repetitions 3 --top-k 5
```

It records per-run retrieval, generation, and total latency, `top_k`, whether generation completed, and a safe error category; summaries include average/median retrieval, generation, and total latency and minimum/maximum total latency. The ignored report is `data/benchmarks/rag_benchmark.json`. **A valid `DEEPSEEK_API_KEY` is required**, and running the benchmark makes provider API requests. No live DeepSeek API smoke test or real RAG benchmark has been completed; no result is inferred from the retrieval-only benchmark.

## Week 4 - NLP Analysis and Filtered Retrieval

### LDA Topic Modeling

`src/topic_modeling.py` fits scikit-learn batch Latent Dirichlet Allocation on the cleaned corpus `text` column. `CountVectorizer` removes English and common news boilerplate stop words, excludes one-character and numeric-only tokens, and caps vocabulary size. The source corpus is read-only. The default is 10 topics with fixed random seed 42; configure `--n-topics`, `--random-state`, `--max-iter`, `--max-features`, and `--top-words`.

```powershell
python src/topic_modeling.py
python src/topic_modeling.py --n-topics 12 --random-state 42 --max-iter 10 --top-words 10
```

Outputs under Git-ignored `data/processed/`:

| File | Schema | Description |
|---|---|---|
| `topic_assignments.parquet` | `document_id` (string), `topic_id` (integer), `topic_probability` (float) | One row per document; probability is the assigned topic's fitted probability. Topic `-1` and probability `0.0` indicate no usable terms. |
| `topic_summary.parquet` | `topic_id` (integer), `top_words` (string) | Representative high-weight words for each topic. |
| `topic_model_manifest.json` | `version`, `vectorizer_version`, `corpus_sha256`, `config`, `document_count` | Cache identity; matching valid outputs are reused unless `--force` is specified. |

If fewer documents have usable terms than requested topics, the topic count is reduced to that document count. If none have usable terms, all assignments are unassigned and the summary is empty.

### Sentiment Analysis and Manual Evaluation

`src/sentiment_analysis.py` uses the local VADER lexicon from pinned `vaderSentiment==3.3.2`; it uses no paid API and downloads no model at runtime. The compound score ranges from -1 to 1. Scores at or above `0.05` are positive; scores at or below `-0.05` are negative; values between them are neutral.

```powershell
python src/sentiment_analysis.py
```

The output `data/processed/sentiment_scores.parquet` has `document_id`, `sentiment_label` (`positive`, `neutral`, or `negative`), and `sentiment_score`. Missing or empty text is neutral with score `0.0`; short text is scored normally. The command also writes `data/benchmarks/sentiment_evaluation.json`.

The version-controlled `data/labels/manual_sentiment_labels.csv` contains 60 manually assigned labels from the clean corpus, balanced at 20 per class. Labels were assigned by reviewing articles independently of VADER predictions. Positive means the central outcome is beneficial, successful, or an achievement; negative means harmful, a loss, failure, or setback; neutral means factual, mixed, or without clear overall polarity. Topic/category alone is not sentiment. This curated set is not a random sample and does not estimate corpus-wide performance.

Measured with VADER 3.3.2 and the thresholds above:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.615 | 0.800 | 0.696 | 20 |
| Neutral | 0.500 | 0.050 | 0.091 | 20 |
| Positive | 0.531 | 0.850 | 0.654 | 20 |

Accuracy is **0.567**. Confusion matrix rows are true labels and columns are predicted labels, ordered negative, neutral, positive: `[[16, 0, 4], [8, 1, 11], [2, 1, 17]]`. VADER is a general-purpose lexicon method; it can miss jargon, entity context, sarcasm, and mixed outcomes. Its score is not a calibrated probability, and its measured neutral recall is low.

### ChromaDB Metadata Enrichment

After topic and sentiment outputs exist, enrich the existing `rag_chunks` collection:

```powershell
python scripts/enrich_chroma_metadata.py
```

The script joins document-level outputs to chunks by `document_id` and attaches scalar `topic_id`, `topic_probability`, `sentiment_label`, and `sentiment_score` where available. Topic ID `-1` is unassigned. Metadata-only `collection.update` preserves chunk IDs, text, embeddings, `document_id`, `source`, and other existing metadata; embeddings are not regenerated.

Missing metadata files or document rows are skipped and reported; unmatched document IDs are counted. Duplicate IDs and invalid schemas or scalar values stop the update. Repeating with unchanged metadata is idempotent. The ignored coverage report is `data/benchmarks/chroma_metadata_enrichment.json`. The validated collection has 5,053 records, with metadata mapped from 4,998 documents.

### Filtered Semantic Retrieval

`src/retrieval.py` supports optional `topic_id` and `sentiment_label` equality filters independently or together. Combined filters use AND. Missing metadata does not match; no matching records returns an empty list. Omitting both filters preserves unfiltered behavior. The result schema, cosine-distance semantics, and `top_k` behavior are unchanged.

```powershell
# Unfiltered
python src/retrieval.py "oil prices and the stock market" --top-k 3

# Topic only, sentiment only, or both
python src/retrieval.py "oil prices and the stock market" --top-k 3 --topic-id 4
python src/retrieval.py "oil prices and the stock market" --sentiment-label negative
python src/retrieval.py "oil prices and the stock market" --topic-id 4 --sentiment-label negative
```

The Python API accepts the same optional arguments, for example `RAGPipeline.answer_query(query, top_k=5, topic_id=4, sentiment_label="negative")`. Existing callers need not pass filters. Filter support has been validated, but no improvement in relevance has been established.

### NLP Validation and Measured Results

Reproduce topic review evidence, sentiment edge probes and evaluation, and live filter-conformance checks with:

```powershell
python scripts/validate_nlp_metadata.py
```

The command writes Git-ignored `data/benchmarks/nlp_metadata_validation.json`. It records the top words and three highest-probability documents per topic, manual topic assessments, edge-case sentiment results, a fresh manual-label evaluation, and four retrieval checks against `rag_chunks`. It uses the existing sentence-transformer and stored vectors; it does not regenerate document embeddings. Checks establish filter conformance, not improved relevance or answer quality.

Manual review of topic top words and three representative documents per topic found:

| Topic | Assessment | Observation |
|---:|---|---|
| 0 | Mixed | Political terms, but representatives include baseball and archaeology. |
| 1 | Mixed | Windows/security cluster with a political-news outlier. |
| 2 | Difficult | Music, trials, batteries, and sports words span unrelated representatives. |
| 3 | Mixed | Iraq/Najaf stories, while Olympic terms also enter the topic. |
| 4 | Difficult | Oil/stock words do not consistently fit sampled arts, currency, and city articles. |
| 5 | Mixed | Earnings terms fit some examples, but a celebrity wedding is also prominent. |
| 6 | Difficult | Hurricane words conflict with representative baseball, climate, and soccer articles. |
| 7 | Mixed | Broad company/court/event terms span drug-import policy and aircraft trade. |
| 8 | Coherent | Broad technology/business cluster around Google, IPOs, search, and stocks. |
| 9 | Coherent | Space/science cluster includes China spaceflight and a Japanese lunar probe, with a software outlier. |

These are qualitative observations, not formal coherence scores. Several topics remain mixed or difficult to interpret.

Current sentiment probes returned `Win!` as positive (`0.6239`); empty and missing text as neutral (`0.0`); technical API/OAuth jargon as neutral (`0.0`); an opposing Microsoft-loss/Apple-profit example as positive (`0.6486`); and a factual committee sentence as neutral (`0.0`). The named-entity example illustrates that compound sentiment can obscure which entity or claim carries the polarity.

On the live collection, unfiltered, topic-only (`topic_id=9`), sentiment-only (`negative`), and combined (`topic_id=8`, `positive`) retrieval each returned five results at `top_k=5`; every result satisfied its requested filter. These observations are conformance checks only, not evidence of better relevance.

## Week 5 - FastAPI Application, Access Logging, and Safe Diagnostics (Phases 1, 2, and 5)

Run the API from the project root:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

`GET /health` returns `{"status":"ok"}` without initializing the RAG pipeline. The first `POST /query` request creates the pipeline; the existing pipeline still initializes retrieval and generation resources lazily as needed.

Example request:

```json
{
	"query": "What happened in the news?",
	"top_k": 5,
	"topic_id": 8,
	"sentiment_label": "positive"
}
```

`query` must contain non-whitespace text. `top_k` is optional (default `5`); `topic_id` and `sentiment_label` are optional and use the same validation as the existing pipeline. The response contains the `RAGResult` fields: `query`, `answer`, `retrieved_chunks`, `context_chunks`, retrieval/generation/total latency, and `failure_kind`. Invalid request bodies receive a generic 422 response; unavailable pipeline initialization returns 503; known upstream failures map to 502, 503, or 504; unexpected query errors receive a generic 500 response. Request logs include method, endpoint, outcome, status, duration, and safe failure category, but not query text or retrieved content. Unexpected errors include server-side traceback frames without exception message text. Live DeepSeek generation still requires `DEEPSEEK_API_KEY`.

## Week 5 - Retrieval Evaluation (Phase 3)

The version-controlled `data/labels/retrieval_evaluation_queries.json` contains nine manually reviewed queries and relevant clean-corpus `document_id` labels (eight scored queries and one explicitly unjudged query). Labels were selected by reviewing corpus article text; they are small positive-relevance judgments, not automatically generated labels or exhaustive assessments of every corpus document.

Run evaluation against the existing `rag_chunks` collection:

```powershell
python scripts/evaluate_retrieval.py --k-values 1 3 5 10
```

The ignored JSON report is `data/benchmarks/retrieval_evaluation.json`. It records dataset/query counts, K values and retriever configuration, per-query metrics and retrieved document IDs, plus macro aggregate metrics.

The evaluation unit is a **document**. Ranked retrieval results are chunks; each chunk is matched using its `document_id`. Duplicate chunks for one document occupy their ranked positions but contribute at most one relevant hit per cutoff. Precision@K is unique relevant documents in the first K chunk positions divided by fixed K, even when fewer than K chunks are returned. Recall@K is unique relevant documents found divided by the number of unique manually labeled relevant document IDs. Aggregates are macro means across scored queries.

An empty `relevant_document_ids` list means no positive relevance judgments are available: the query receives `null` metrics and is excluded from aggregates, rather than being treated as a verified query with zero relevant documents. Unlisted documents in scored queries count as nonrelevant for this run, so incomplete judgments can understate precision. These metrics evaluate this small labeled set only; they do not establish improvement over another retriever or prove corpus-wide quality.

The latest run contains 9 queries: 8 scored, 1 unjudged, and 47 relevant query/document judgments. Macro Precision@K and Recall@K were:

| K | Macro Precision@K | Macro Recall@K |
|---:|---:|---:|
| 1 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 |
| 5 | 0.0250 | 0.0179 |
| 10 | 0.0875 | 0.1515 |

## Week 5 - End-to-End RAG Evaluation (Phase 4)

Run the existing manually judged queries through `RAGPipeline`:

```powershell
python scripts/evaluate_rag.py --top-k 5
```

The script reuses the query and relevant document IDs from `data/labels/retrieval_evaluation_queries.json`; it does not create reference answers or modify retrieval/generation behavior. It writes the Git-ignored `data/benchmarks/rag_evaluation.json`, containing per-case answers, retrieved/context document IDs, `failure_kind`, latency fields, answer-generation status, category-level failure counts, and aggregate latency statistics.

The answer-related measure is `relevant_context_recall_proxy`: the fraction of manually relevant document IDs present in `RAGResult.context_chunks`. It measures evidence availability to the generator, not semantic answer correctness. No exact-match or text-overlap score is used. Review each recorded answer against its source documents for correctness and grounding.

The latest local run had **9 cases: 0 successful and 9 failed**, all with `missing_api_key`; no answers were generated. Mean/median end-to-end latency was `0.0439` / `0.0229` seconds, measured on these failed runs; these values are not successful answer-generation latency. The relevant-context recall proxy averaged `0.0179` over the 8 judged cases; the unjudged case is excluded. This proxy and run do not establish answer quality or retrieval improvement. A valid `DEEPSEEK_API_KEY` is required to evaluate generated answers; without it, the report records safe pipeline failures.

## Week 5 - API, Evaluation, and Reproducibility

### Week 5 Overview

Week 5 adds a FastAPI service around the existing RAG pipeline, query-free API access logging, safe server-side diagnostics for unexpected retrieval and generation errors, a manually judged retrieval evaluation, and an end-to-end RAG evaluation.

The implementation reuses `RAGPipeline` rather than duplicating retrieval, prompt construction, or generation, keeping API behavior aligned with the existing command-line pipeline. Pipeline initialization is lazy, so checking `/health` does not load the embedding model or initialize the vector database. Access logs record request metadata but omit query text and retrieved content. Error responses are generic for unexpected failures, while known pipeline failures map to HTTP statuses without exposing provider details. Retrieval metrics evaluate unique relevant documents but keep a fixed K precision denominator so duplicate chunks consume rank positions. Queries without positive relevance labels are marked unjudged and excluded from aggregate metrics rather than counted as zero.

Week 5 files:

```text
src/api.py
scripts/evaluate_retrieval.py
scripts/evaluate_rag.py
data/labels/retrieval_evaluation_queries.json
tests/test_api.py
tests/test_evaluate_retrieval.py
tests/test_evaluate_rag.py
```

The test count, project structure, and reproduction steps in this section supersede earlier project-structure and reproduction details; the 229-test count supersedes the earlier count of 188.

### FastAPI Application, Access Logging, and Safe Diagnostics

From the project root, start the local API:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

`GET /health` returns `{"status":"ok"}` without initializing the RAG pipeline. The first `POST /query` creates the pipeline; its retriever and generation client remain lazy as in the existing implementation. FastAPI's interactive API documentation is available at `http://127.0.0.1:8000/docs` with the default application configuration.

The query request accepts a non-whitespace `query`, optional positive integer `top_k` (default `5`), optional non-negative integer `topic_id`, and optional `sentiment_label` (`negative`, `neutral`, or `positive`). The optional filters are passed to the existing pipeline. For example:

```json
{
  "query": "What happened in the news?",
  "top_k": 5,
  "topic_id": 8,
  "sentiment_label": "positive"
}
```

The response preserves the `RAGResult` fields: `query`, `answer`, `retrieved_chunks`, `context_chunks`, `retrieval_latency_seconds`, `generation_latency_seconds`, `total_latency_seconds`, and `failure_kind`.

| HTTP status | Meaning |
|---:|---|
| 200 | Query completed without a pipeline failure. |
| 422 | Invalid request body; the client receives a generic validation message. |
| 502 | Known upstream authentication, connection, response, output-limit, API, or generation failure. |
| 503 | Pipeline initialization unavailable, retrieval resources unavailable, retrieval failure, or unavailable/rate-limited LLM service. |
| 504 | Upstream LLM request timed out. |
| 500 | Unexpected query error or an unrecognized pipeline failure; the client receives a generic message. |

Access records contain the HTTP method, endpoint, outcome, status, duration, and safe failure category. They do not contain query text, prompts, retrieved text, answers, or credentials. For unexpected exceptions, server-side diagnostics contain the operation, exception type, and traceback frame locations, without exception messages or source lines.

With the service running, send a query from PowerShell using `curl.exe` (rather than PowerShell's `curl` alias):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"What happened in the news?","top_k":5}'
```

Live answer generation requires a valid `DEEPSEEK_API_KEY`, configured in the process environment or local `.env` file. The API has no authentication or rate limiting and is intended for local use, not public deployment. On macOS/Linux, activate the environment and copy the example configuration with `source .venv/bin/activate` and `cp .env.example .env`; add credentials locally and do not commit `.env`.

### Retrieval Evaluation

`data/labels/retrieval_evaluation_queries.json` contains 9 manually reviewed queries: 8 scored queries, 1 explicitly unjudged query, and 47 relevant query/document judgments. These are small, positive-relevance judgments selected by reviewing article text, not exhaustive labels for the corpus.

Run the evaluation against the existing `rag_chunks` collection:

```powershell
python scripts/evaluate_retrieval.py --k-values 1 3 5 10
```

The script writes the Git-ignored report `data/benchmarks/retrieval_evaluation.json`. It records the dataset and query counts, retriever configuration, per-query relevance labels and retrieved document IDs, per-query metrics, and macro aggregate metrics.

The evaluation unit is a document, although retrieval returns ranked chunks. Each chunk is mapped to a document through `document_id`; duplicate chunks for one document occupy their rank positions but that document contributes at most one relevant hit at each cutoff. Precision@K is the number of unique relevant documents in the first K chunk positions divided by fixed K, even if fewer than K chunks were returned. Recall@K is the number of unique relevant documents found divided by the number of labeled relevant documents. Macro metrics are the arithmetic means over scored queries. An empty `relevant_document_ids` list produces null metrics and is excluded from aggregates.

The rerun for this submission measured:

| K | Macro Precision@K | Macro Recall@K |
|---:|---:|---:|
| 1 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 |
| 5 | 0.0250 | 0.0179 |
| 10 | 0.0875 | 0.1515 |

Retrieval quality is weak on this labeled set. Possible contributing factors are hypotheses, not tested explanations: the positive-only labels are incomplete and can understate precision; 500-character chunks may split useful context; `all-MiniLM-L6-v2` is a general-purpose rather than domain-tuned embedding model; and the collection represents a 5,000-record training subset rather than the full news domain. The measurements do not demonstrate improvement over another retriever or establish corpus-wide quality.

### End-to-End RAG Evaluation

Run the existing labeled queries through `RAGPipeline`:

```powershell
python scripts/evaluate_rag.py --top-k 5
```

The script reuses the queries and relevant document IDs from `data/labels/retrieval_evaluation_queries.json`; it does not create reference answers. It writes the Git-ignored `data/benchmarks/rag_evaluation.json`, which records per-case answers, retrieved and context document IDs, `failure_kind`, latency fields, generation status, failure counts, and aggregate latency statistics.

`relevant_context_recall_proxy` is the fraction of manually relevant document IDs present in `RAGResult.context_chunks`. It measures evidence availability to the generator, not semantic answer correctness. The evaluation does not use exact-match or text-overlap answer scoring; any generated answer must be reviewed manually against its source documents.

The latest saved local report contains **9 cases: 0 successful and 9 failed**, all with `missing_api_key`; no answers were generated. Mean/median end-to-end latency was **0.0439 / 0.0229 seconds** on failed runs only, not successful generation latency. The context-recall proxy averaged **0.0179** over 8 judged cases; the unjudged query is excluded. These results do not establish answer quality. To obtain live generation results, configure a valid `DEEPSEEK_API_KEY` locally and rerun `python scripts/evaluate_rag.py --top-k 5`; this makes provider API requests.

### Week 5 Testing and Verification

Run the focused API, pipeline, and evaluation tests, followed by the full suite, from the project root:

Activate the project's `.venv` first, as shown in the setup steps below. The focused command invokes its interpreter explicitly.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py tests\test_rag_pipeline.py tests\test_evaluate_retrieval.py tests\test_evaluate_rag.py
python -m pytest -q
```

The measured focused result was **69 passed**; the measured full-suite result was **229 passed in 7.87 seconds**. Tests use local fixtures, fake models, or mocked clients and do not make network or DeepSeek calls.

### Updated Reproduction Steps

From a clean clone, run these commands in order from the project root in PowerShell. The data download and model setup may require network access:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_env.py
python scripts/download_data.py
python src/preprocessing.py
python scripts/validate_dataset.py
python src/chunking.py
python src/embeddings.py
python src/vector_store.py
python src/topic_modeling.py
python src/sentiment_analysis.py
python scripts/enrich_chroma_metadata.py
python scripts/validate_nlp_metadata.py
python scripts/evaluate_retrieval.py --k-values 1 3 5 10
python -m pytest -q
```

To serve the API after building the local data and vector collection, run this in a separate terminal:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

End-to-end RAG evaluation is optional and requires a valid locally configured `DEEPSEEK_API_KEY`:

```powershell
python scripts/evaluate_rag.py --top-k 5
```

On macOS/Linux, activate the virtual environment with `source .venv/bin/activate` and copy the local environment template with `cp .env.example .env`; use `python` for the commands above after activation.

### Week 5 Known Limitations

- Retrieval metrics are weak on this small labeled set: Precision@K / Recall@K are 0.0000 / 0.0000 at K=1 and K=3, 0.0250 / 0.0179 at K=5, and 0.0875 / 0.1515 at K=10.
- Manual relevance labels are positive-only and incomplete, so measured precision may be understated.
- No successful live DeepSeek generation, live RAG benchmark, or live RAG evaluation has been completed. The context-recall proxy measures evidence availability, not answer quality.
- The API has no authentication, rate limiting, or deployment hardening; it is intended for local use.
- Earlier limitations still apply: LDA topics are partly mixed or difficult to interpret; VADER's manual evaluation is curated, with low neutral recall; and metadata-filter conformance does not demonstrate improved relevance or answer quality.

## Testing, Final Verification, Reproduction, and Known Limitations

### Tests and Final Verification

Run the focused API, pipeline, and evaluation tests, then the full test suite, from the project root using the existing Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py tests\test_rag_pipeline.py tests\test_evaluate_retrieval.py tests\test_evaluate_rag.py
.\.venv\Scripts\python.exe -m pytest -q
```

The suite covers preprocessing, chunking, embeddings, vector-store ingestion and metadata preservation, retrieval filters and both evaluation metrics, topic modeling, sentiment scoring/evaluation, NLP validation, benchmarks, and mocked LLM behavior. Tests use local fixtures or fake models and do not download the corpus, call DeepSeek, or run live benchmarks. The latest full-suite validation passed **229 tests**.

Other checks:

```powershell
python scripts/verify_env.py
python scripts/validate_dataset.py
```

The validated artifacts contain 4,998 clean documents, 5,053 chunks, 5,053 embeddings of dimension 384, and 5,053 `rag_chunks` records. Generated datasets, vector-store files, and benchmark reports are Git-ignored.

### Full Pipeline Reproduction

Required local pipeline steps, in dependency order, from a clean clone:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_env.py
python scripts/download_data.py
python src/preprocessing.py
python scripts/validate_dataset.py
python src/chunking.py
python src/embeddings.py
python src/vector_store.py
python src/topic_modeling.py
python src/sentiment_analysis.py
python scripts/enrich_chroma_metadata.py
python scripts/validate_nlp_metadata.py
python scripts/evaluate_retrieval.py --k-values 1 3 5 10
python -m pytest -q
```

The first embedding or retrieval run may download and cache the sentence-transformer model. Dataset, Parquet, ChromaDB, and benchmark artifacts are local generated files and are not committed. The manual sentiment labels are version-controlled and refer to document IDs from the acquired clean corpus.

Optional checks and benchmarks:

```powershell
python scripts/benchmark_retrieval.py
python scripts/benchmark_rag.py --repetitions 3 --top-k 5
python scripts/evaluate_rag.py --top-k 5
```

The retrieval-only benchmark needs the corpus, vectors, model, and Chroma collection. The RAG benchmark and end-to-end RAG evaluation require a valid `DEEPSEEK_API_KEY` to generate answers and make provider API requests. Neither a live RAG benchmark nor a successful live-LLM evaluation has been completed in this workspace.

### Known Limitations

- LDA topics are partly mixed or difficult to interpret; the topic assessments are qualitative, not formal coherence scores.
- VADER's 60-example evaluation is curated rather than randomly sampled, so its results are not an unbiased estimate of corpus-wide performance; neutral recall is low on this set.
- Sentiment scores are not calibrated confidence, and lexical scoring can mishandle jargon, named entities, sarcasm, and mixed outcomes.
- Metadata filters were checked for conformance on the existing collection; filtered retrieval has not been shown to improve relevance or answer quality.
