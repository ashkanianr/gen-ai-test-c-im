# Manulife SmartClaim Intelligence

A production-quality GenAI system for automated insurance claim processing using Retrieval-Augmented Generation (RAG). The system analyzes insurance claims against policy documents to make APPROVE/REJECT/ESCALATE decisions with full traceability and reliability scoring.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Design](#system-design)
- [Core Components](#core-components)
- [Evaluation Methodology](#evaluation-methodology)
- [Reliability & Safety Mechanisms](#reliability--safety-mechanisms)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

## Architecture Overview

The system follows a clean, model-agnostic architecture with clear separation of concerns:

```
┌─────────────────┐
│  Policy PDF     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  PDF Loader     │────▶│  Chunking     │────▶│ Embeddings  │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │  FAISS Store │
                                            └──────┬───────┘
                                                   │
┌─────────────────┐                              │
│  Claim Input    │                              │
└────────┬────────┘                              │
         │                                        │
         ▼                                        ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  RAG Pipeline  │────▶│  Retrieval   │────▶│   LLM       │
└────────┬────────┘     └──────────────┘     └──────┬──────┘
         │                                            │
         ▼                                            ▼
┌─────────────────┐                          ┌─────────────┐
│  Decision       │                          │  Citations  │
│  (APPROVE/      │                          └─────────────┘
│   REJECT/       │
│   ESCALATE)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │
│  Layer          │
└─────────────────┘
```

## System Design

### Design Principles

1. **Model Agnosticism**: All LLM interactions go through a unified `LLMClient` interface. Vendor-specific logic (Gemini, OpenRouter) is isolated to `llm_client.py`.

2. **Clean Architecture**: Business logic is separated from infrastructure concerns. Vector DB access is abstracted behind `retriever.py`, allowing easy swaps (FAISS → Pinecone, Weaviate, etc.).

3. **Evaluation First**: Evaluation is not an afterthought—it's integrated into the decision pipeline and runs automatically to ensure reliability.

4. **Traceability**: Every decision includes cited policy sections, retrieved chunks, and confidence scores for full auditability.

5. **Safety by Design**: Low-confidence decisions automatically escalate to human review. System prompts explicitly forbid external knowledge.

### RAG Design Choices

**Chunking Strategy**:
- Uses LangChain's `RecursiveCharacterTextSplitter` with configurable chunk size (default: 1000 chars) and overlap (default: 200 chars)
- Overlap ensures context continuity across chunk boundaries
- Metadata (section_id, page_number, policy_name) attached to each chunk for traceability

**Retrieval Strategy**:
- FAISS with cosine similarity for fast, in-memory retrieval
- Top-K retrieval (default: 5 chunks) balances context richness with token limits
- Normalized embeddings ensure accurate cosine similarity calculations

**Reasoning Strategy**:
- LLM receives only retrieved policy chunks as context
- System prompt explicitly forbids external knowledge
- JSON-structured output for consistent parsing
- Auto-escalation when confidence < 0.75

## Core Components

### `app/llm_client.py`
Model-agnostic LLM interface supporting Gemini and OpenRouter. All LLM calls route through this abstraction.

### `app/embeddings.py`
Embedding service wrapper for Gemini embeddings. Provides batch processing and dimension information.

### `app/pdf_loader.py`
PDF parsing using `pdfplumber` (primary) or `pypdf` (fallback). Extracts text with page information, chunks with overlap, and attaches metadata.

### `app/retriever.py`
Vector database abstraction layer. Currently implements FAISS but designed for easy swaps to cloud vector DBs.

### `app/rag_pipeline.py`
Main orchestration component:
- `ingest_policy()`: PDF → chunks → embeddings → vector store
- `process_claim()`: Claim → retrieval → reasoning → decision

### `evaluation/`
Three-layer evaluation system:
- **Retrieval Evaluation**: Measures if required policy sections were retrieved
- **Faithfulness Evaluation**: LLM-as-judge verifies all statements are policy-grounded
- **Decision Accuracy**: Compares predicted vs expected decisions

## Evaluation Methodology

The system implements comprehensive evaluation across three dimensions:

### 1. Retrieval Recall
**Metric**: Percentage of expected policy sections found in retrieved chunks

**Purpose**: Ensures the retrieval system finds relevant policy information

**Calculation**:
```
recall = (found_sections / expected_sections) × 100%
```

### 2. Faithfulness Score
**Metric**: LLM-as-judge evaluation (0.0-1.0) of whether all explanation statements are supported by policy text

**Purpose**: Detects hallucinations and unsupported claims

**Method**: Separate LLM evaluation using `judge_faithfulness.txt` prompt template

### 3. Decision Accuracy
**Metric**: Percentage of correct decisions (APPROVE/REJECT/ESCALATE)

**Purpose**: Measures end-to-end system correctness

### Composite Confidence Score

The system calculates a weighted composite confidence score:

```
confidence = (0.4 × retrieval_recall) + 
             (0.4 × faithfulness_score) + 
             (0.2 × decision_correctness)
```

**Escalation Threshold**: If confidence < 0.75, decision automatically changes to ESCALATE

## Reliability & Safety Mechanisms

### 1. Hallucination Control
- **System Prompts**: Explicitly forbid external knowledge
- **Context Limitation**: LLM receives only retrieved policy chunks
- **Refusal Mechanism**: System must respond with ESCALATE if policy text is insufficient
- **Faithfulness Evaluation**: LLM-as-judge verifies every statement

### 2. Confidence-Based Escalation
- **Multi-Factor Scoring**: Combines retrieval quality, faithfulness, and decision correctness
- **Automatic Escalation**: Low confidence (< 0.75) triggers ESCALATE decision
- **Transparency**: Confidence scores included in all responses

### 3. Traceability
- **Cited Sections**: Every decision includes exact policy section references
- **Retrieved Chunks**: Full list of retrieved chunks with similarity scores
- **Raw Output**: Original LLM response preserved for audit

### 4. Error Handling
- **Graceful Degradation**: Missing API keys, network errors handled gracefully
- **Fallback Providers**: OpenRouter fallback if Gemini unavailable
- **Validation**: Input validation and type checking throughout

## Installation

### Prerequisites
- Python 3.8+
- Gemini API key (or OpenRouter API key for fallback)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd manulife-smartclaim
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. **Convert sample policies to PDFs** (optional):
```bash
# If you have reportlab installed
python scripts/create_sample_pdfs.py

# Or manually convert data/policies/*.txt to PDFs
```

## Usage

### CLI Interface

**Ingest a policy**:
```bash
python -m app.main ingest data/policies/health_policy.pdf
```

**Process a claim**:
```bash
python -m app.main claim "I need to file a claim for a broken arm from a skiing accident" --policy-path data/policies/health_policy.pdf
```

**Process a claim with JSON output**:
```bash
python -m app.main claim "Claim text here" --policy-path data/policies/health_policy.pdf --json
```

**Run evaluation**:
```bash
python -m app.main evaluate data/evaluation_set.json --policy-paths data/policies/health_policy.pdf data/policies/travel_policy.pdf --output evaluation_report.json
```

### FastAPI Server

**Start the server**:
```bash
python -m app.main serve --host 0.0.0.0 --port 8000
```

**API Endpoints**:

- `POST /ingest-policy`: Upload and process a policy PDF
  ```bash
  curl -X POST "http://localhost:8000/ingest-policy" \
    -F "file=@data/policies/health_policy.pdf"
  ```

- `POST /process-claim`: Process an insurance claim
  ```bash
  curl -X POST "http://localhost:8000/process-claim" \
    -H "Content-Type: application/json" \
    -d '{"claim_text": "I need to file a claim for a broken arm"}'
  ```

- `GET /health`: Health check
  ```bash
  curl http://localhost:8000/health
  ```

- `GET /policies`: List ingested policies
  ```bash
  curl http://localhost:8000/policies
  ```

### Programmatic Usage

```python
from app.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest policy
pipeline.ingest_policy("data/policies/health_policy.pdf")

# Process claim
result = pipeline.process_claim("I need to file a claim for a broken arm")

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Explanation: {result['explanation']}")
```

## Limitations and Future Improvements

### Current Limitations

1. **In-Memory Vector Store**: FAISS is in-memory, so policies are lost on restart. Production would require persistent storage (Pinecone, Weaviate, etc.).

2. **Simple Chunking**: Current chunking doesn't respect document structure (headers, sections). Future: semantic chunking or structure-aware splitting.

3. **Basic Section Detection**: Section IDs are auto-generated. Future: parse actual section headers from PDFs.

4. **Single Policy at a Time**: System processes one policy per claim. Future: multi-policy support with policy selection logic.

5. **Evaluation Dataset**: Synthetic dataset may not reflect real-world complexity. Future: real anonymized claims dataset.

### Future Improvements

1. **Persistent Vector Store**: Migrate to cloud vector DB (Pinecone, Weaviate) for production scalability

2. **Advanced Chunking**: Implement semantic chunking or hierarchical document structure parsing

3. **Multi-Modal Support**: Handle images, tables, and diagrams in policy PDFs

4. **Fine-Tuned Models**: Fine-tune models on insurance domain data for better accuracy

5. **Real-Time Monitoring**: Add logging, metrics, and monitoring for production deployment

6. **A/B Testing Framework**: Compare different retrieval strategies, chunk sizes, and LLM models

7. **Human-in-the-Loop**: Integrate with human review workflows for escalated claims

8. **Multi-Language Support**: Extend to handle policies and claims in multiple languages

## Project Structure

```
manulife-smartclaim/
├── app/
│   ├── __init__.py
│   ├── main.py                 # CLI and FastAPI entry point
│   ├── rag_pipeline.py         # Main orchestration
│   ├── retriever.py            # Vector DB abstraction
│   ├── llm_client.py           # Model-agnostic LLM interface
│   ├── embeddings.py           # Embedding service
│   ├── pdf_loader.py           # PDF parsing and chunking
│   └── prompts/
│       ├── claim_decision.txt
│       └── judge_faithfulness.txt
├── evaluation/
│   ├── __init__.py
│   ├── retrieval_eval.py       # Retrieval recall metrics
│   ├── faithfulness_eval.py    # LLM-as-judge evaluation
│   ├── decision_eval.py        # Decision accuracy
│   └── run_evaluation.py       # End-to-end evaluation
├── data/
│   ├── policies/               # Sample policy PDFs
│   ├── claims/                 # Sample claim documents
│   └── evaluation_set.json     # Evaluation dataset
├── scripts/
│   └── create_sample_pdfs.py   # Utility to convert text to PDF
├── requirements.txt
├── .env.example
└── README.md
```

## License

This project is a take-home assessment and is not intended for production use without proper security review, compliance checks, and additional testing.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.
