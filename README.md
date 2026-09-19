# RAG Knowledge Extraction Project

This project establishes the foundation for a retrieval-augmented generation (RAG) knowledge extraction workflow. It will eventually acquire source documents, clean and structure their text, create embeddings, and make the resulting knowledge searchable.

## Week 1 Scope

Week 1 establishes a reproducible Python environment and the repository layout for the project. The planned deliverable is a verified environment, a modular text-cleaning pipeline with unit tests, and a validated clean dataset.

This foundation phase intentionally does not:

- Download or ingest the 5,000+ document dataset.
- Implement the text preprocessing pipeline.
- Add preprocessing unit tests.

## Setup

Create and activate a virtual environment from the `RAG Knowledge Extractor Project/` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To use the environment from Command Prompt instead:

```bat
.venv\Scripts\activate
```

No dataset download is required for this foundation phase. Dataset files belong in `data/raw/` and generated outputs belong in `data/processed/`; their contents are ignored by Git.

## Project Structure

```text
RAG Knowledge Extractor Project/
├── data/
│   ├── raw/             # Source datasets; contents are Git-ignored
│   └── processed/       # Cleaned/validated datasets; contents are Git-ignored
├── src/                 # Project source modules
├── tests/               # Unit and validation tests
├── scripts/             # Repeatable project utilities and checks
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Ingestion

Placeholder for the future dataset acquisition and ingestion workflow. This section will document source selection, download procedures, input formats, provenance, and validation checks.

## Preprocessing

Placeholder for the future modular text-cleaning pipeline. This section will document normalization, filtering, chunking, output schemas, and the unit-test strategy.
