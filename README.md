# Project Overview

This repository contains Week 1 of a RAG-powered knowledge extraction system. The overall goal is to acquire real-world text, clean and validate it, and prepare a reliable corpus for later retrieval-augmented generation work.

## Week 1 Scope

Week 1 covers:

- Python environment setup
- Environment verification
- Raw dataset acquisition
- Modular text preprocessing
- Unit testing
- Clean dataset validation

## Project Structure

```text
RAG Knowledge Extractor Project/
├── data/
│   ├── raw/             # Git-ignored source corpus
│   └── processed/       # Git-ignored cleaned corpus
├── src/                 # Reusable preprocessing modules
├── tests/               # Fast pytest unit tests
├── scripts/             # Environment, acquisition, and validation scripts
├── .gitignore
├── requirements.txt
└── README.md
```

`data/raw/` stores the acquired source documents. `data/processed/` stores generated clean datasets. Both dataset directories are Git-ignored so large data files are not committed. `src/` contains the preprocessing implementation, `tests/` contains its unit tests, and `scripts/` contains repeatable project utilities.

## Environment Setup

From the project root, create and activate a Python virtual environment, then install the listed dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt, activate the same environment with:

```bat
.venv\Scripts\activate
```

`requirements.txt` currently lists the required packages without pinned version numbers.

## Environment Verification

Run the environment check from the project root:

```powershell
python scripts/verify_env.py
```

The script reports the Python version, checks whether PyTorch is installed, reports CUDA and GPU availability when present, and tests imports for pandas, sentence-transformers, ChromaDB, and spaCy. Missing required imports produce a non-zero exit code. CUDA or GPU absence is reported as informational because the project can run on CPU.

## Dataset Acquisition

The project uses the publicly accessible AG News dataset from Hugging Face (`fancyzhx/ag_news`). One AG News article is treated as one document. The acquisition script selected the first 5,000 records from the training split and preserved document ID, source, split, label, source identifier, title field, and raw text.

Acquire or validate the raw corpus with:

```powershell
python scripts/download_data.py
```

The script writes UTF-8 JSON Lines data to:

```text
data/raw/ag_news_train_5000.jsonl
```

It prints progress while downloading and reuses an existing valid output instead of downloading it again. Raw data is Git-ignored because datasets can be large and should be reproducibly acquired rather than stored in repository history.

## Preprocessing

The modular pipeline in `src/preprocessing.py` applies these deterministic stages in order:

1. HTML stripping and character-reference decoding
2. Unicode NFC normalization
3. Excessive whitespace cleanup
4. English-language filtering using `langdetect`

Null and non-string text values are handled safely. Source metadata is preserved, while the `text` field is replaced with its cleaned value. Run the pipeline with:

```powershell
python src/preprocessing.py
```

## Output Dataset

The pipeline writes the clean dataset to:

```text
data/processed/clean_corpus.parquet
```

Statistics from the actual Week 1 run:

```text
Raw documents: 5,000
Cleaned documents: 4,998
Removed documents: 2
Empty documents removed: 0
Language-filtered documents: 2
Average text length before cleaning: 246.3 characters
Average text length after cleaning: 244.8 characters
```

The separate validation command checks the generated file, schema, document IDs, empty text values, and basic text statistics:

```powershell
python scripts/validate_dataset.py
```

The validated output contains 4,998 documents, seven columns, unique non-null document IDs, no empty text values, and text lengths ranging from 89 to 959 characters.

## Testing

Run the unit tests from the project root:

```powershell
pytest
```

The 18 tests cover HTML stripping, Unicode normalization, whitespace cleanup, language filtering, null and non-string inputs, combined cleaning behavior, record filtering, metadata preservation, and Parquet schema output. Tests use small in-memory examples and do not download the full corpus.

## Reproducibility

From a clean clone, run the following commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_env.py
python scripts/download_data.py
python src/preprocessing.py
pytest
python scripts/validate_dataset.py
```

The acquisition script obtains the AG News training data through the Hugging Face `datasets` library, writes the reproducible 5,000-document raw JSONL corpus, and skips acquisition when the valid output already exists. The preprocessing command then regenerates the ignored Parquet output.

## Current Limitations / Future Weeks

The completed Week 1 implementation does not include:

- Text chunking
- Embedding generation
- ChromaDB indexing or other vector database work
- Semantic retrieval
- LLM response generation
- RAG evaluation, including precision, recall, or hallucination detection
- Topic modeling or sentiment analysis
- FastAPI or another serving API

These capabilities are outside the completed Week 1 scope.
