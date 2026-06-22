# Nuclear Compliance RAG System

A production-grade Retrieval Augmented Generation (RAG) system for querying nuclear regulatory documents using natural language. Built with LangChain, ChromaDB, and Anthropic Claude, with a custom evaluation harness for measuring answer quality.

## Problem Statement

Nuclear regulatory compliance requires engineers to navigate dense, technical documents like NRC Regulatory Guides, ASME standards, and DOE guidance. Finding precise answers across multiple documents is time-consuming and error-prone. This system allows compliance engineers to ask natural language questions and receive cited, grounded answers pulled directly from authoritative source documents.

## Project Background

This is a public rebuild of a RAG system originally developed using proprietary nuclear industry documents. The original system demonstrated the same core architecture against internal compliance documentation. The original system was designed around a local LLM architecture to ensure sensitive compliance documents remained within the department. This version has been reconstructed using publicly available NRC Regulatory Guides sourced from the NRC's ADAMS database and the Anthropic Claude API to allow for open sharing and portfolio demonstration while respecting document confidentiality.

## Architecture

The system is built around two pipelines:

**Ingestion Pipeline**
1. PDF documents are loaded from the `data/` directory
2. Text is extracted page by page using PyMuPDF, preserving page metadata for citations
3. Text is split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter (700 character chunks, 100 character overlap)
4. Chunks are embedded and stored in a persistent ChromaDB vector store

**Query Pipeline**
1. User submits a natural language question
2. The question is embedded using the same model as ingestion
3. ChromaDB returns the top-k most semantically similar chunks
4. Retrieved chunks and the original question are passed to Claude with a grounding prompt
5. Claude generates a cited answer using only the provided context
6. Sources are displayed alongside the answer with document name and page number

## Tech Stack

| Tool | Purpose | Why |
|------|---------|-----|
| Anthropic Claude | LLM backbone | Strong instruction following for grounded, cited answers |
| ChromaDB | Vector store | Simple setup, persistent local storage, no external service required |
| LangChain | Text splitting and chain orchestration | Well-tested chunking utilities |
| PyMuPDF | PDF text extraction | Fast and accurate extraction with page-level metadata |
| Streamlit | UI | Rapid prototyping with built-in chat components |
| Docker | Containerization | Reproducible deployment across environments |

## Project Structure
nuclear-rag/

├── data/                  # PDF documents (gitignored)

├── src/

│   ├── ingest.py          # Ingestion pipeline

│   ├── retriever.py       # Semantic retrieval from ChromaDB

│   └── chain.py           # LLM generation with grounding prompt

├── tests/

│   ├── eval.py            # Custom evaluation harness

│   └── eval_results.json  # Latest eval scores

├── app.py                 # Streamlit UI

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

## Getting Started

### Prerequisites

- Python 3.11+
- An Anthropic API key
- Nuclear regulatory PDF documents (place in `data/` directory)
  - NRC Regulatory Guide 1.140 Rev. 3 and 1.52 Rev. 4 have been included and are publicly available at
    https://adams-search.nrc.gov/home

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/JARandle/Nuclear-RAG
cd nuclear-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
    ```bash
    ANTHROPIC_API_KEY=your_api_key_here
    ```

5. Add your PDF documents to the `data/` directory, then run ingestion:
```bash
python -m src.ingest
```

6. Launch the app:
```bash
streamlit run app.py
```

### Running with Docker

1. Add your PDF documents to the `data/` directory and ensure your `.env` file is present.

2. Build and run:
```bash
docker-compose up --build
```

3. Open `http://localhost:8501` in your browser.

The `data/` and `.chroma/` directories are mounted as volumes so your documents and vector store persist between container runs without being baked into the image.

## Example Questions

These questions were tested against NRC Regulatory Guide 1.52 and NRC Regulatory Guide 1.140:

**Q: What are the in-place leak testing acceptance criteria for HEPA filter systems?**

> Combined penetration and leakage must be less than 0.05% of the challenge aerosol at system-rated flow ±10%, warranting 99% removal efficiency for particulates. [Source: NRC Regulatory Guide 1.52, Page 12]

**Q: What challenge aerosol is used for HEPA filter penetration testing?**

> Dioctyl phthalate (DOP) or 4-centistoke polyalpha olefin (PAO) are acceptable challenge aerosols per ASME AG-1-2009 Section TA. [Source: NRC Regulatory Guide 1.52, Page 11]

**Q: What is the maximum flow rate for a single HEPA filtration unit?**

> A single filtration unit is limited to 850 m³/min (30,000 CFM) to ensure reliable in-place testing. Systems requiring higher capacity should use multiple parallel units. [Source: NRC Regulatory Guide 1.140, Page 8]

## Evaluation

The system includes a custom Claude-as-judge evaluation harness that scores answers on two dimensions:

- **Faithfulness**: Does the answer only use information from the retrieved context?
- **Relevancy**: Does the answer actually address what was asked?

Each dimension is scored 1 to 5 by Claude acting as an expert judge, then normalized to a 0 to 1 scale.

### Running Evals

```bash
python -m tests.eval
```

Results are saved to `tests/eval_results.json` and displayed in the app sidebar.

### Current Scores

| Metric | Score |
|--------|-------|
| Faithfulness | 0.95 |
| Relevancy | 0.90 |

Scores are based on 5 test questions covering HEPA filter testing criteria, challenge aerosol specifications, flow rate limits, testing trigger conditions, and quality assurance requirements.

## Design Decisions

**Chunk size of 700 characters with 100 character overlap**
Nuclear regulatory documents use dense, precise language where a single paragraph can contain multiple distinct requirements. A 700 character chunk size keeps related requirements together while staying within embedding model limits. The 100 character overlap prevents requirements from being split across chunk boundaries.

**Grounding prompt with explicit citation requirement**
The system prompt explicitly instructs Claude to answer only from the provided context and to cite sources by document and page number. If the context does not contain enough information to answer, Claude is instructed to say so rather than guess. 

**Custom evaluation harness**
Rather than using an off-the-shelf evaluation library, this project implements a lightweight Claude-as-judge harness. This approach has several advantages: no additional dependencies, full transparency into scoring criteria, and the ability to tailor scoring rubrics to the nuclear compliance domain. The judge prompt was designed specifically to evaluate whether answers are appropriate for a regulated technical environment. The evaluation harness can also be modified with additional parameters to test the project as thoroughly as needed.

**Document-agnostic architecture**
The ingestion pipeline treats the `data/` directory as a plug-and-play document store. Any PDF can be ingested without code changes, making the system transferable to other regulatory domains such as environmental compliance, medical device standards, or legal documentation. Publicly available NRC regulatory documents are present for demonstration purposes and may be removed if needed.

**ChromaDB with persistent local storage**
ChromaDB's persistent client stores the vector store to disk rather than memory, meaning documents only need to be ingested once. The `.chroma/` directory is mounted as a Docker volume so the vector store survives container restarts without re-ingestion.

## Future Improvements

- Hybrid search combining semantic and keyword retrieval for improved recall on specific regulatory codes and section numbers
- Metadata filtering to scope queries to a specific document or standard
- Streaming responses for improved perceived latency
- Web-based document upload interface for adding new standards without CLI access