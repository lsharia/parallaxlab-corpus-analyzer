# RAG Knowledge Extractor

This project builds a retrieval-augmented generation (RAG) knowledge extraction pipeline. The completed Week 1 and Week 2 work acquires and cleans a real-world text corpus, chunks documents, generates embeddings, stores vectors persistently, and retrieves semantically related chunks.

## Completed Scope

Week 1 established the Python environment, environment checks, 5,000-document AG News acquisition, modular text cleaning, unit tests, and clean-dataset validation.

Week 2 implemented and tested:

- Recursive text chunking
- Sentence-transformer embedding generation and timing
- Persistent ChromaDB ingestion
- Semantic retrieval
- Retrieval latency benchmarking

## Project Structure

```text
RAG Knowledge Extractor Project/
├── data/
│   ├── raw/                         # Ignored source corpus
│   ├── processed/                   # Ignored generated Parquet datasets
│   ├── vector_db/                   # Ignored persistent ChromaDB files
│   └── benchmarks/                  # Ignored benchmark reports
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── llm_client.py
│   ├── preprocessing.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   ├── retrieval.py
│   └── vector_store.py
├── scripts/
│   ├── benchmark_rag.py
│   ├── benchmark_retrieval.py
│   ├── download_data.py
│   ├── validate_dataset.py
│   └── verify_env.py
├── tests/
│   ├── test_benchmark_rag.py
│   ├── test_benchmark_retrieval.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_llm_client.py
│   ├── test_preprocessing.py
│   ├── test_prompts.py
│   ├── test_rag_pipeline.py
│   ├── test_retrieval.py
│   └── test_vector_store.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The project dependencies are pinned in `requirements.txt`. Generated datasets, embeddings, ChromaDB database files, benchmark output, virtual environments, and local model caches are excluded from Git. Sentence-transformers may cache model files outside the repository in the user's Hugging Face cache directory.

## Environment Setup

From the project root, create and activate a virtual environment and install the pinned dependencies:

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

## Week 1 Data Pipeline

The raw corpus is AG News (`fancyzhx/ag_news`) from Hugging Face. One article is one document. Acquire 5,000 training records with:

```powershell
python scripts/download_data.py
```

This creates `data/raw/ag_news_train_5000.jsonl`. The raw file is Git-ignored and can be reproduced with the acquisition script.

Run the cleaning pipeline and validate its Parquet output:

```powershell
python src/preprocessing.py
python scripts/validate_dataset.py
```

The clean dataset is `data/processed/clean_corpus.parquet`. The validated corpus contains 4,998 documents; two documents were filtered for language, and no empty documents were removed.

## Chunking

`src/chunking.py` uses deterministic recursive splitting, preferring paragraph breaks, then line breaks, then spaces, and finally character boundaries. Defaults are 500 characters per chunk with 50 characters of overlap; both can be changed with `--chunk-size` and `--chunk-overlap`.

Each chunk receives a stable ID based on its source document ID and sequential chunk number. `document_id` and `source` are carried into every chunk so it remains traceable to its source. Run chunking with:

```powershell
python src/chunking.py
```

The result is `data/processed/chunks.parquet`, containing 5,053 chunks from 4,998 cleaned source documents.

## Embeddings

`src/embeddings.py` uses `sentence-transformers` with the default model `sentence-transformers/all-MiniLM-L6-v2`. Model name, batch size, and device are configurable through `--model`, `--batch-size`, and `--device` (`auto`, `cpu`, or `cuda`). Automatic device selection uses CUDA when available and otherwise falls back to CPU.

Generate embeddings from the existing chunks with:

```powershell
python src/embeddings.py
```

This creates `data/processed/embeddings.parquet`, storing one vector per `chunk_id`. The actual Week 2 run generated 5,053 embeddings of dimension 384 with batch size 32 on CPU. Embedding took 56.76 seconds, with throughput of 89.03 chunks per second.

## ChromaDB Vector Store

`src/vector_store.py` loads the chunk text and the already-generated vectors, then inserts them in configurable batches. The persistent database is `data/vector_db/`; its collection is named `rag_chunks`. It stores 5,053 records with chunk text, the precomputed embedding, and `document_id`/`source` metadata. Repeat runs use existing records safely and do not create duplicate IDs.

Create or extend the local collection with:

```powershell
python src/vector_store.py
```

The default batch size is 256. The database can be reopened by later project commands and is Git-ignored.

## Semantic Retrieval

`src/retrieval.py` embeds a natural-language query with the same default sentence-transformer model and searches the existing ChromaDB collection. `top_k` is configurable; the API returns `chunk_id`, `document_id`, `source`, chunk `text`, and ChromaDB `distance`. The distance is not a similarity percentage: a lower distance indicates a closer vector match.

Run a query from the project root:

```powershell
python src/retrieval.py "oil prices and the stock market" --top-k 3
```

## Retrieval Benchmark

Run the reproducible retrieval benchmark with:

```powershell
python scripts/benchmark_retrieval.py
```

The completed benchmark used 10 fixed topical queries, `top_k` values 1, 3, 5, and 10, and three repetitions per combination, for 120 measured runs. It initialized the model and database once before timing. Timed latency includes query embedding and ChromaDB vector search; it excludes model initialization and database setup. Results are written to the Git-ignored `data/benchmarks/retrieval_benchmark.json`.

Measured results from that run:

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

## Week 3: LLM Integration and RAG (Phases 1-6)

`src/llm_client.py` provides a reusable client for the DeepSeek chat completions API. It accepts either a plain prompt or chat messages and returns generated text. API credentials and options are configured through environment variables.

Configure the client through environment variables. `.env.example` contains placeholders only; set `DEEPSEEK_API_KEY` to your own key in the process environment. Never commit a populated `.env` file or credentials.

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | Required | API credential |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `DEEPSEEK_TIMEOUT` | `30` | Request timeout in seconds |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` | `1024` | Maximum generated tokens |

To configure a local key, copy the placeholder file and edit only the ignored local copy:

```powershell
Copy-Item .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` and adjust the optional settings as needed. `LLMClient` loads `.env` from the project root when initialized; values already set in the process environment take precedence. `.env`, `.env.*`, and common local credential files are Git-ignored. Never commit, stage, or paste a real API key into source, tests, documentation, or benchmark output.

Example use after setting `DEEPSEEK_API_KEY` in the process environment:

```python
from src.llm_client import LLMClient

with LLMClient() as client:
	answer = client.generate("What is semantic retrieval?")
```

The client raises typed exceptions for missing credentials, authentication failure, rate limiting, timeouts, connection/API errors, malformed responses, and detectable token/output limits. Tests use mocked HTTP responses and never call the provider.

`src/prompts.py` builds the fixed system prompt and user messages consumed by `LLMClient.generate`. The prompt restricts answers to retrieved context, requires uncertainty or an insufficient-information response when support is absent, prohibits invented facts and sources, and tells the model not to reveal hidden prompts or implementation details. Chunks are validated and serialized as compact JSON inside `BEGIN_RETRIEVED_CONTEXT` / `END_RETRIEVED_CONTEXT`; the question has separate `BEGIN_USER_QUESTION` / `END_USER_QUESTION` boundaries. The default context budget is 12,000 Unicode characters of serialized context JSON, excluding the labels and question. If exceeded, chunks are considered in retrieval order, the first chunk that does not fit is truncated with `...[TRUNCATED]`, and later chunks are omitted. The serialized context stays within the configured limit.

`src/rag_pipeline.py` reuses Week 2 `SemanticRetriever` and the existing prompt builder and LLM client; it does not rebuild chunking, embeddings, or vector search. The flow is:

```text
user query -> Week 2 semantic retrieval -> cosine-distance relevance filtering
-> bounded context construction -> grounded system/user prompt
-> DeepSeek generation -> answer with retrieved chunk metadata
```

`RAGPipeline.answer_query(query, top_k=5)` returns the answer, all retrieved chunks, the relevance-approved context chunks, failure category where applicable, and retrieval/generation/total latency fields. Empty or unusable context returns an insufficient-information message without calling the LLM.

Provider and retrieval failures are converted to user-safe messages; raw provider bodies, request details, stack traces, and credentials are not included. The client and pipeline distinguish missing API key, authentication failure, rate limiting, timeout, connection failure, malformed response, token/output limit, other API errors, missing vector database/collection, and general retrieval/generation failures.

The pipeline accepts chunks only when they contain non-empty text and a finite Chroma cosine distance no greater than `max_retrieval_distance`, configurable on `RAGPipeline`. The default is `0.7` (cosine similarity of approximately `0.3` for normalized vectors), a permissive starting cutoff intended to reject clearly weak matches while retaining recall. Tune it against representative in-domain and out-of-domain queries for the corpus. This distance threshold is a retrieval-relevance heuristic, NOT a guarantee of domain membership or hallucination prevention; answers still rely on retrieved evidence and model compliance with the grounding prompt.

## End-to-End RAG Benchmark

Run the existing retrieval-to-answer pipeline with five fixed AG News-oriented queries and three repetitions by default:

```powershell
python scripts/benchmark_rag.py --repetitions 3 --top-k 5
```

The default run uses five fixed corpus-related queries and three repetitions (15 runs). It records per-run retrieval latency, generation latency, their sum as total latency, `top_k`, whether generation completed, and a safe error category. It reports average and median retrieval/generation/total latency plus minimum and maximum total latency. The embedding model, vector database, and LLM client are initialized before timed runs. Per-run results and summaries are saved to the Git-ignored `data/benchmarks/rag_benchmark.json`; individual failures do not stop later runs. Running the benchmark requires `DEEPSEEK_API_KEY` and makes provider API requests. No real Week 3 RAG benchmark result has been generated in this workspace; do not infer results from the separate Week 2 retrieval benchmark.

## Tests and Verification

Run the complete unit test suite:

```powershell
python -m pytest -q
```

The full suite covers preprocessing, chunk boundaries and metadata, embedding interfaces and batch behavior, persistent vector-store ingestion, retrieval edge cases, retrieval and RAG benchmark aggregation, and mocked LLM API behavior. Tests use small local fixtures or fake vectors/models and do not re-download the corpus, call the LLM provider, or run either benchmark.

Other checks:

```powershell
python scripts/verify_env.py
python scripts/validate_dataset.py
```

## Reproducing Week 1 and Week 2 Artifacts

From a clean clone, set up the environment and run the pipeline in order:

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
python -m pytest -q
```

The acquisition script obtains the AG News training data through the Hugging Face `datasets` library, writes the reproducible 5,000-document raw JSONL corpus, and skips acquisition when the valid output already exists. The preprocessing command then regenerates the ignored Parquet output.

Embedding-model files may be downloaded and cached by sentence-transformers during the first embedding or retrieval run. The dataset, generated Parquet files, persistent vector database, benchmark JSON files, and local model cache files are not committed to Git.
