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
│   ├── preprocessing.py
│   ├── retrieval.py
│   └── vector_store.py
├── scripts/
│   ├── benchmark_retrieval.py
│   ├── download_data.py
│   ├── validate_dataset.py
│   └── verify_env.py
├── tests/
│   ├── test_benchmark_retrieval.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   └── test_vector_store.py
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

## Tests and Verification

Run the complete unit test suite:

```powershell
python -m pytest -q
```

The full suite currently contains 68 passing tests. Tests cover preprocessing, chunk boundaries and metadata, embedding interfaces and batch behavior, persistent vector-store ingestion, retrieval edge cases, and benchmark result aggregation. Tests use small local fixtures or fake vectors/models and do not re-download the corpus or run the full benchmark.

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

Embedding-model files may be downloaded and cached by sentence-transformers during the first embedding or retrieval run. The dataset, generated Parquet files, persistent vector database, benchmark JSON, and local model cache files are not committed to Git.

## Current Limitations

The completed implementation stops at retrieval and latency benchmarking. It does not include an LLM, answer generation, a RAG generation pipeline, retrieval quality evaluation, or an API service such as FastAPI.
