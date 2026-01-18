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

The system separates policy ingestion, runtime inference, and evaluation concerns. Evaluation is not a single step, but a layered mechanism consisting of runtime safety checks (faithfulness and confidence) and offline quality measurement (retrieval recall and decision accuracy).

```
┌─────────────────┐
│  Policy PDF     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  PDF Loader     │────▶│  Chunking     │────▶│ Embeddings  │
│                 │     │  + Metadata  │     │ (RETRIEVAL_ │
│                 │     │  (section_id, │     │  DOCUMENT)  │
│                 │     │   page_num,   │     └──────┬──────┘
│                 │     │   policy_name)│             │
└─────────────────┘     └──────────────┘             │
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │ Embedding     │
                                            │ Cache         │
                                            │ (.cache/)     │
                                            └──────┬───────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │  FAISS Store │
                                            │  (Vector DB) │
                                            └──────┬───────┘
                                                   │
┌─────────────────┐                              │
│  Claim Input    │                              │
└────────┬────────┘                              │
         │                                        │
         ▼                                        ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  RAG Pipeline  │────▶│  Retrieval           │────▶│  LLM             │
│                 │     │  (Top-K + similarity │     │  (Policy-grounded│
│                 │     │   score + metadata) │     │   reasoning only)│
│                 │     │                      │     │  Must rely ONLY │
│                 │     │  Metadata flow:      │     │  on retrieved   │
│                 │     │  • section_id        │     │  policy chunks  │
│                 │     │  • page_number       │     │  No external    │
│                 │     │  • policy_name       │     │  knowledge      │
│                 │     │  • similarity_score  │     │  If insufficient│
└────────┬────────┘     └──────────────────────┘     │  → ESCALATE    │
         │                  │                          └──────┬───────────┘
         │                  │                                   │
         │                  │                                   │
         │                  └───────────┐                       │
         │                              │                       │
         ▼                              ▼                       ▼
┌─────────────────┐              ┌─────────────┐      ┌─────────────┐
│  Decision       │              │  Citations   │      │  Metadata   │
│  (APPROVE/      │              │  (extracted   │      │  (section_id,│
│   REJECT/       │              │   from chunk │      │   page_num)  │
│   ESCALATE)     │              │   metadata)  │      └─────────────┘
└────────┬────────┘              └─────────────┘
         │
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Runtime Evaluation (Production Safety)                 │
│  • Faithfulness check (LLM-as-judge)                    │
│  • Retrieval confidence (similarity scores)              │
│  • Composite confidence scoring                          │
│  → ESCALATE if confidence < 0.75                        │
└─────────────────────────────────────────────────────────┘
         │
         │ (runs during inference, affects decision)
         │
         ▼
    [Production Decision Output]

┌─────────────────────────────────────────────────────────┐
│  Offline Evaluation (Model Quality)                    │
│  • Retrieval Recall (on synthetic dataset)             │
│  • Decision Accuracy (predicted vs expected)            │
│  • Aggregate metrics & reports                           │
│  → Used for model comparison, prompt tuning, testing    │
└─────────────────────────────────────────────────────────┘
         │
         │ (runs outside production on evaluation datasets)
         │
         ▼
    [Evaluation Reports]
```

## System Design

### Design Principles

The system separates policy ingestion, runtime inference, and evaluation concerns. Evaluation is not a single step, but a layered mechanism consisting of runtime safety checks (faithfulness and confidence) and offline quality measurement (retrieval recall and decision accuracy).

1. **Model Agnosticism**: All LLM interactions go through a unified `LLMClient` interface. Vendor-specific logic (Gemini, OpenRouter) is isolated to `llm_client.py`.

2. **Clean Architecture**: Business logic is separated from infrastructure concerns. Vector DB access is abstracted behind `retriever.py`, allowing easy swaps (FAISS → Pinecone, Weaviate, etc.).

3. **Layered Evaluation Architecture**:
   - **Runtime Evaluation** (Production Safety): Integrated into `rag_pipeline.py` during inference
     * Faithfulness checks (LLM-as-judge verifies policy grounding)
     * Retrieval confidence (similarity scores from top-K chunks)
     * Composite confidence scoring
     * Automatic escalation (confidence < 0.75 → force ESCALATE)
     * **Purpose**: Prevent hallucinations and unsafe approvals in production
   
   - **Offline Evaluation** (Model Quality): Separate `evaluation/` module runs on synthetic datasets
     * Retrieval recall (measures if required sections were retrieved)
     * Decision accuracy (predicted vs expected on evaluation set)
     * Aggregate metrics and reports
     * **Purpose**: Model comparison, prompt tuning, regression testing

4. **Policy-Only Grounding Constraint**: 
   - LLM reasoning must rely ONLY on retrieved policy chunks
   - System prompts explicitly forbid external knowledge
   - If retrieved text is insufficient → model must refuse or escalate (never guess)
   - Enforced by: prompt design, faithfulness evaluation, confidence-based escalation

5. **Traceability**: Every decision includes cited policy sections (from chunk metadata), retrieved chunks with similarity scores, and confidence scores for full auditability.

6. **Safety by Design**: Low-confidence decisions automatically escalate to human review. System prompts explicitly forbid external knowledge.

### RAG Design Choices

**Chunking Strategy**:
- Uses LangChain's `RecursiveCharacterTextSplitter` with configurable chunk size (default: 1000 chars) and overlap (default: 200 chars)
- Overlap ensures context continuity across chunk boundaries
- Metadata (section_id, page_number, policy_name) attached to each chunk for traceability

**Embedding Caching**:
- Policy embeddings are automatically cached to `.cache/embeddings/` directory
- Cache key includes: file path, modification time, file size, chunk size, and chunk overlap
- Cache is automatically invalidated when policy files change
- Significantly reduces API calls and prevents rate limits during development and testing
- First ingestion generates and caches embeddings; subsequent runs load from cache (no API calls)

**Retrieval Strategy**:
- FAISS with cosine similarity for fast, in-memory retrieval
- Top-K retrieval (default: 5 chunks) balances context richness with token limits
- Normalized embeddings ensure accurate cosine similarity calculations
- **Retrieval Quality Metrics**: Each retrieved chunk includes:
  - Similarity score (cosine similarity, 0-1)
  - Metadata (policy section ID, page number, policy name)
  - Full chunk text for LLM context
- Retrieved chunks with metadata are passed to the LLM for reasoning
- Citations are extracted from chunk metadata (section_id, page_number)

**Reasoning Strategy (Policy-Only Grounding)**:
- **Strict Constraint**: LLM reasoning must rely ONLY on retrieved policy chunks. System prompts explicitly forbid external knowledge. If insufficient information, the model must respond with ESCALATE (never guess or use general knowledge).
- **No External Knowledge**: System prompts explicitly forbid the use of external knowledge, general insurance domain knowledge, or training data knowledge
- **Insufficient Information Handling**: If retrieved policy text is insufficient to make a decision, the model must refuse or escalate (never guess)
- **Enforcement Mechanisms**:
  - **Prompt Design**: Explicitly states "You MUST base your decision ONLY on the policy text provided. Do NOT use any external knowledge."
  - **Runtime Faithfulness Check**: LLM-as-judge verifies all statements are policy-grounded during inference
  - **Confidence-Based Escalation**: Low confidence automatically triggers ESCALATE, preventing unsafe approvals
- JSON-structured output for consistent parsing
- Auto-escalation when confidence < 0.75

## Core Components

### `app/llm_client.py`
Model-agnostic LLM interface supporting Gemini and OpenRouter. All LLM calls route through this abstraction.

### `app/embeddings.py`
Embedding service wrapper for Gemini embeddings. Provides batch processing and dimension information.

### `app/embedding_cache.py`
Embedding cache manager that stores policy embeddings to disk, avoiding regeneration for unchanged policies. Reduces API calls and prevents rate limits. Automatically invalidates cache when policies change (based on file modification time and content hash).

### `app/pdf_loader.py`
PDF parsing using `pdfplumber` (primary) or `pypdf` (fallback). Extracts text with page information, chunks with overlap, and attaches metadata.

### `app/retriever.py`
Vector database abstraction layer. Currently implements FAISS but designed for easy swaps to cloud vector DBs.

### `app/rag_pipeline.py`
Main orchestration component:
- `ingest_policy()`: PDF → chunks → embeddings (with caching) → vector store
- `process_claim()`: Claim → retrieval → reasoning → decision

**Embedding Caching**: The pipeline automatically caches policy embeddings in `.cache/embeddings/` to avoid regenerating embeddings for unchanged policies. Cache is keyed by file path, modification time, and chunking parameters, ensuring automatic invalidation when policies change.

### `evaluation/`
Two-mode evaluation system:

**Runtime Evaluation** (integrated in `rag_pipeline.py`):
- **Faithfulness Evaluation**: LLM-as-judge verifies all explanation statements are supported by policy text (runs during inference)
- **Retrieval Confidence**: Calculates average similarity scores from retrieved chunks
- **Composite Confidence**: Weighted combination of retrieval quality and faithfulness
- **Automatic Escalation**: Forces ESCALATE decision if confidence < 0.75

**Offline Evaluation** (runs on synthetic datasets):
- **Retrieval Evaluation**: Measures if required policy sections were retrieved (recall metric)
- **Decision Accuracy**: Compares predicted vs expected decisions on evaluation dataset
- **Aggregate Metrics**: Generates reports for model comparison and regression testing

## Evaluation Methodology

The system implements a two-mode evaluation architecture that separates production safety from model quality measurement.

### Runtime Evaluation (Production Safety)

Runtime evaluation occurs **during inference** and directly affects the decision. It prevents hallucinations and unsafe approvals in production.

#### 1. Faithfulness Check
**Metric**: LLM-as-judge evaluation (0.0-1.0) of whether all explanation statements are supported by policy text

**Purpose**: Detects hallucinations and unsupported claims in real-time

**Method**: Separate LLM evaluation using `judge_faithfulness.txt` prompt template that verifies every statement in the decision explanation is traceable to retrieved policy chunks

**Behavior**: If faithfulness score is low, confidence is reduced, potentially triggering escalation

#### 2. Retrieval Confidence
**Metric**: Average similarity score of retrieved chunks (0.0-1.0)

**Purpose**: Measures retrieval quality for the current claim

**Calculation**: Average of cosine similarity scores from top-K retrieved chunks

**Behavior**: Low retrieval confidence indicates insufficient or irrelevant policy context

#### 3. Composite Confidence Score

The system calculates a weighted composite confidence score **during inference**:

```
confidence = (0.4 × avg_retrieval_similarity) + 
             (0.4 × faithfulness_score) + 
             (0.2 × llm_confidence_indicator)
```

**Components**:
- `avg_retrieval_similarity`: Average cosine similarity scores from top-K retrieved chunks (0-1)
- `faithfulness_score`: LLM-as-judge evaluation of policy grounding (0-1)
- `llm_confidence_indicator`: Model-reported uncertainty or refusal markers, normalized to [0,1]. Derived from LLM response confidence signals (e.g., "HIGH", "MEDIUM", "LOW" mapped to 0.9, 0.7, 0.5). In production, this can be replaced with calibrated confidence estimation from model logits or uncertainty quantification methods.

**Escalation Threshold**: If confidence < 0.75, decision **automatically changes to ESCALATE**

**Production Impact**: This runtime check ensures no low-confidence decisions are approved or rejected without human review. Note: This uses model confidence signals, not decision correctness (which is only measured in offline evaluation).

### Offline Evaluation (Model Quality)

Offline evaluation occurs **outside production** on synthetic datasets. It measures system quality for model comparison, prompt tuning, and regression testing.

#### 1. Retrieval Recall
**Metric**: Percentage of expected policy sections found in retrieved chunks

**Purpose**: Ensures the retrieval system finds relevant policy information

**Calculation**:
```
recall = (found_sections / expected_sections) × 100%
```

**Usage**: Used to compare different retrieval strategies, chunk sizes, and embedding models

#### 2. Decision Accuracy
**Metric**: Percentage of correct decisions (APPROVE/REJECT/ESCALATE) on evaluation dataset

**Purpose**: Measures end-to-end system correctness

**Usage**: Tracks model performance over time, identifies regression, guides prompt improvements

#### 3. Aggregate Metrics

Offline evaluation generates comprehensive reports including:
- Per-category accuracy (clearly covered, clearly excluded, ambiguous claims)
- Confusion matrices
- Retrieval recall distribution
- Faithfulness score distribution
- Escalation rate analysis

**Usage**: Model comparison, A/B testing, regression testing, compliance reporting

## Reliability & Safety Mechanisms

### 1. Policy-Only Grounding Constraint

**Core Principle**: The LLM is **forbidden from using external knowledge**. All reasoning must be based solely on retrieved policy chunks.

**Enforcement Mechanisms**:

- **System Prompts**: Explicitly state "You MUST base your decision ONLY on the policy text provided. Do NOT use any external knowledge."
- **Context Limitation**: LLM receives only retrieved policy chunks as context—no general knowledge, no domain expertise, no training data
- **Refusal Mechanism**: System prompt requires "If the policy text does not contain sufficient information to make a decision, you MUST respond with ESCALATE"
- **Runtime Faithfulness Check**: LLM-as-judge verifies every statement in the explanation is traceable to policy text
- **Confidence-Based Escalation**: Low confidence automatically triggers ESCALATE, preventing unsafe decisions

**Traceability**: Every decision includes:
- Cited policy sections (section_id, page_number)
- Retrieved chunks with similarity scores
- Raw model output for audit

This constraint ensures regulatory compliance and prevents the system from making decisions based on assumptions or general knowledge.

### 2. Confidence-Based Escalation
- **Multi-Factor Scoring**: Combines retrieval quality, faithfulness, and model confidence signals (not decision correctness, which is only measured offline)
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

## Non-Goals

This system is designed with clear boundaries to ensure appropriate use and governance:

- **Does not replace human adjudication**: The system provides decision recommendations with confidence scores. Final decisions require human review, especially for escalated cases.

- **Does not learn from production decisions**: The system does not update its model or prompts based on production outcomes. All improvements are made through offline evaluation and controlled updates.

- **Does not perform probabilistic risk scoring**: The system focuses on policy compliance and coverage determination, not actuarial risk assessment or premium calculation.

These boundaries ensure the system remains a decision-support tool rather than an autonomous decision-maker, aligning with regulated-AI best practices for insurance applications.

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

### Embedding Cache

The system automatically caches policy embeddings to reduce API calls and avoid rate limits:

- **Cache Location**: `.cache/embeddings/` (automatically created)
- **Cache Behavior**: 
  - First ingestion: Generates embeddings and saves to cache
  - Subsequent runs: Loads from cache (no API calls)
  - Policy changes: Automatically detects changes and regenerates
- **Cache Management**: Cache is keyed by file path, modification time, and chunking parameters
- **Benefits**: 
  - Reduces embedding API calls by ~90% for repeated policy ingestion
  - Prevents rate limit issues during development and testing
  - Faster policy ingestion on subsequent runs

**Note**: The cache directory (`.cache/`) is excluded from git via `.gitignore`. To clear cache, simply delete the `.cache/` directory.

## Usage

### CLI Interface

**Ingest a policy** (embeddings are automatically cached):
```bash
python -m app.main ingest data/policies/health_policy.pdf
# First run: Generates embeddings and caches them (saves to .cache/embeddings/)
# Subsequent runs: Loads from cache (much faster, no API calls)
```

**Process a claim** (using sample claim file):
```bash
python -m app.main claim data/claims/health_claim_001_approve.txt --policy-path data/policies/health_policy.txt
```

**Process a claim with JSON output**:
```bash
python -m app.main claim data/claims/health_claim_001_approve.txt --policy-path data/policies/health_policy.txt --json
```

**Run test suite** (test multiple scenarios):
```bash
python scripts/test_claims.py
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

1. **In-Memory Vector Store**: FAISS is in-memory, so policies are lost on restart. Production would require persistent storage (Pinecone, Weaviate, etc.). **Note**: Embeddings are cached to disk, but the FAISS index itself is rebuilt on restart.

2. **Simple Chunking**: Current chunking doesn't respect document structure (headers, sections). Future: semantic chunking or structure-aware splitting.

3. **Basic Section Detection**: Section IDs are auto-generated. Future: parse actual section headers from PDFs.

4. **Single Policy at a Time**: System processes one policy per claim. Future: multi-policy support with policy selection logic.

5. **Evaluation Dataset**: Synthetic dataset may not reflect real-world complexity. Future: real anonymized claims dataset.

6. **API Rate Limits**: Free tier Gemini API has rate limits (100 embedding requests/min, 20 generation requests/min). The embedding cache helps significantly, but high-volume testing may still hit generation limits.

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
│   ├── embedding_cache.py      # Embedding cache manager
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
│   ├── create_sample_pdfs.py   # Utility to convert text to PDF
│   └── test_claims.py          # Automated test suite for sample claims
├── .cache/                     # Embedding cache (auto-generated, git-ignored)
│   └── embeddings/             # Cached policy embeddings
├── requirements.txt
├── .env.example
└── README.md
```

## License

This project is a take-home assessment and is not intended for production use without proper security review, compliance checks, and additional testing.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.
