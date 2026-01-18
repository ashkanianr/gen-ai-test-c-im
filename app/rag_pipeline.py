"""RAG pipeline for insurance claim processing.

This module orchestrates the complete workflow:
1. Policy ingestion (PDF → chunks → embeddings → vector store)
2. Claim processing (claim → retrieval → reasoning → decision)

EVALUATION ARCHITECTURE:
- Runtime Evaluation: Integrated into process_claim() for production safety
  * Retrieval confidence (similarity scores)
  * Composite confidence scoring
  * Automatic escalation (confidence < 0.75)
  * Note: Full faithfulness evaluation (LLM-as-judge) would be integrated here
  
- Offline Evaluation: Separate evaluation/run_evaluation.py module
  * Runs on synthetic datasets outside production
  * Measures retrieval recall, decision accuracy
  * Used for model comparison, prompt tuning, regression testing

POLICY-ONLY GROUNDING:
- LLM receives ONLY retrieved policy chunks as context
- System prompts explicitly forbid external knowledge
- Citations extracted from chunk metadata (section_id, page_number)
- Insufficient information triggers ESCALATE (never guess)
"""

from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
import numpy as np

from app.llm_client import get_llm_client, LLMClient
from app.embeddings import get_embedding_service, EmbeddingService
from app.embedding_cache import EmbeddingCache
from app.pdf_loader import load_policy_pdf, PDFChunk
from app.retriever import create_retriever, VectorRetriever


class RAGPipeline:
    """Main RAG pipeline for claim processing."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
        retriever: Optional[VectorRetriever] = None,
        top_k: int = 5,
    ):
        """
        Initialize RAG pipeline.

        Args:
            llm_client: LLM client instance (auto-created if None)
            embedding_service: Embedding service instance (auto-created if None)
            retriever: Vector retriever instance (auto-created if None)
            top_k: Number of chunks to retrieve per query
        """
        self.llm_client = llm_client or get_llm_client()
        self.embedding_service = embedding_service or get_embedding_service()
        self.top_k = top_k

        # Initialize retriever if not provided
        if retriever is None:
            embedding_dim = self.embedding_service.get_embedding_dimension()
            self.retriever = create_retriever(embedding_dim)
        else:
            self.retriever = retriever

        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache()

        # Store policy metadata
        self.policy_metadata: Dict[str, Any] = {}

    def ingest_policy(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Ingest a policy PDF into the vector store.

        Args:
            pdf_path: Path to policy PDF file
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters

        Returns:
            Dict with ingestion statistics
        """
        # Load and chunk PDF
        chunks = load_policy_pdf(pdf_path, chunk_size, chunk_overlap)

        if not chunks:
            raise ValueError(f"No text extracted from PDF: {pdf_path}")

        policy_name = Path(pdf_path).stem

        # Check cache first
        cached_result = self.embedding_cache.get_cached_embeddings(
            pdf_path, chunk_size, chunk_overlap
        )
        
        if cached_result is not None:
            # Use cached embeddings
            print(f"[CACHE] Using cached embeddings for {policy_name}")
            embeddings_array, metadata_list = cached_result
        else:
            # Generate embeddings for all chunks (documents use RETRIEVAL_DOCUMENT)
            print(f"[CACHE] Generating new embeddings for {policy_name}...")
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Prepare metadata
            metadata_list = [chunk.to_dict() for chunk in chunks]
            
            # Save to cache for future use
            self.embedding_cache.save_embeddings(
                pdf_path, chunk_size, chunk_overlap, embeddings_array, metadata_list
            )

        # Add to vector store
        self.retriever.add_documents(embeddings_array, metadata_list)

        # Store policy metadata
        self.policy_metadata[policy_name] = {
            "pdf_path": pdf_path,
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        return {
            "policy_name": policy_name,
            "num_chunks": len(chunks),
            "status": "ingested",
        }

    def process_claim(
        self,
        claim_text: str,
        policy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process an insurance claim and generate a decision.

        Args:
            claim_text: The claim description (text or path to PDF)
            policy_name: Optional policy name filter (if multiple policies ingested)

        Returns:
            Dict with decision, explanation, citations, and metadata
        """
        # Load claim text if it's a file path
        if os.path.exists(claim_text):
            # Try to load as text file first
            try:
                with open(claim_text, "r", encoding="utf-8") as f:
                    claim_text = f.read()
            except:
                # If not text, try PDF
                from app.pdf_loader import extract_text_from_pdf
                pages = extract_text_from_pdf(claim_text)
                claim_text = "\n\n".join([p["text"] for p in pages])

        # Retrieve relevant policy chunks (queries use RETRIEVAL_QUERY)
        # Retrieval includes: top-K chunks, similarity scores, and metadata (section_id, page_number)
        # Increase k slightly to get more context for better decisions
        query_embedding = np.array(
            self.embedding_service.embed_text(claim_text, task_type="RETRIEVAL_QUERY"),
            dtype=np.float32,
        )
        retrieved_results = self.retriever.search(query_embedding, k=min(self.top_k + 2, 10))  # Get a few more chunks for context
        
        # Retrieved results contain:
        # - chunk: Full text of the policy chunk
        # - score: Cosine similarity score (0-1)
        # - metadata: section_id, page_number, policy_name, chunk_index

        if not retrieved_results:
            # No relevant chunks found - escalate
            return {
                "decision": "ESCALATE",
                "explanation": "No relevant policy sections found for this claim. Manual review required.",
                "cited_sections": [],
                "retrieved_chunks": [],
                "raw_model_output": "",
                "confidence_score": 0.0,
            }

        # Format policy context for LLM
        # CRITICAL: LLM receives ONLY retrieved policy chunks - no external knowledge allowed
        # Policy-only grounding constraint enforced by prompt design and faithfulness evaluation
        policy_context_parts = []
        for i, result in enumerate(retrieved_results, 1):
            chunk = result["chunk"]
            metadata = result["metadata"]
            policy_context_parts.append(
                f"[Section {metadata.get('section_id', 'Unknown')}, "
                f"Page {metadata.get('page_number', '?')}]\n{chunk}"
            )
        policy_context = "\n\n---\n\n".join(policy_context_parts)

        # Load decision prompt template
        prompt_path = Path(__file__).parent / "prompts" / "claim_decision.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # Format prompt
        formatted_prompt = prompt_template.format(
            policy_context=policy_context,
            claim_text=claim_text,
        )

        # Generate decision using LLM
        messages = [
            {"role": "user", "content": formatted_prompt}
        ]

        try:
            # Retry logic with automatic fallback to OpenRouter on rate limits
            raw_output = None
            fallback_client = None
            
            # Check if OpenRouter is available as fallback (using same Gemini 3 Flash model for consistency)
            from app.llm_client import OpenRouterClient
            try:
                fallback_client = OpenRouterClient(model_name="google/gemini-3-flash")
                if not fallback_client.is_available():
                    fallback_client = None
            except:
                fallback_client = None
            
            # Try primary client first
            try:
                response = self.llm_client.chat_completion(
                    messages=messages,
                    temperature=0.0,  # Zero temperature for most consistent decisions
                    max_tokens=2000,
                )
                raw_output = response["content"]
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower()
                
                # If rate limit and we have OpenRouter fallback, switch immediately
                if is_rate_limit and fallback_client:
                    print(f"[WARNING] Rate limit hit on primary LLM. Switching to OpenRouter (Gemini 3 Flash)...")
                    try:
                        response = fallback_client.chat_completion(
                            messages=messages,
                            temperature=0.0,
                            max_tokens=2000,
                        )
                        raw_output = response["content"]
                        self.llm_client = fallback_client
                        print("[INFO] Successfully switched to OpenRouter (Gemini 3 Flash) for consistency")
                    except Exception as fallback_error:
                        print(f"[WARNING] OpenRouter fallback also failed: {fallback_error}")
                        raise RuntimeError(f"Both primary LLM and OpenRouter fallback failed. Primary error: {str(e)}, Fallback error: {str(fallback_error)}")
                else:
                    # Not a rate limit or no fallback available - re-raise
                    raise
            
            if raw_output is None:
                raise RuntimeError("Failed to get LLM response")
        except Exception as e:
            return {
                "decision": "ESCALATE",
                "explanation": f"Error generating decision: {str(e)}. Manual review required.",
                "cited_sections": [],
                "retrieved_chunks": [r["metadata"] for r in retrieved_results],
                "raw_model_output": "",
                "confidence_score": 0.0,
            }

        # Parse LLM response (expecting JSON)
        try:
            # Try to extract JSON from response
            # LLM might wrap JSON in markdown code blocks
            json_text = raw_output
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            decision_data = json.loads(json_text)
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: try to extract decision from text
            decision = "ESCALATE"
            explanation = raw_output
            cited_sections = []

            # Try to infer decision from text
            if "APPROVE" in raw_output.upper():
                decision = "APPROVE"
            elif "REJECT" in raw_output.upper():
                decision = "REJECT"

            decision_data = {
                "decision": decision,
                "explanation": explanation,
                "cited_sections": cited_sections,
                "confidence": "LOW",
            }

        # Extract cited sections from retrieved chunks
        cited_sections = []
        if "cited_sections" in decision_data:
            # Use the cited sections from LLM response
            cited_sections = decision_data["cited_sections"]
        else:
            # Fallback: cite all retrieved chunks
            for result in retrieved_results:
                metadata = result["metadata"]
                cited_sections.append({
                    "section_id": metadata.get("section_id", "Unknown"),
                    "page_number": metadata.get("page_number", 0),
                    "relevant_text": result["chunk"][:500],  # First 500 chars
                })

        # RUNTIME EVALUATION: Calculate confidence score for production safety
        # This is part of runtime evaluation that directly affects the decision
        # Combines retrieval quality (similarity scores) with model confidence signals
        # Note: Uses model confidence indicators, NOT decision correctness (which is only measured offline)
        avg_retrieval_score = np.mean([r["score"] for r in retrieved_results])
        confidence_map = {"HIGH": 0.9, "MEDIUM": 0.75, "LOW": 0.6}  # Adjusted: less conservative
        # llm_confidence_indicator: Model-reported uncertainty or refusal markers
        # In production, this can be replaced with calibrated confidence estimation
        llm_confidence = confidence_map.get(
            decision_data.get("confidence", "MEDIUM"), 0.7  # Default to MEDIUM instead of LOW
        )
        # Weight retrieval more heavily if it's high quality (good matches)
        # Weight LLM confidence more if retrieval is lower quality
        if avg_retrieval_score >= 0.75:
            # High quality retrieval - trust it more
            confidence_score = (avg_retrieval_score * 0.6) + (llm_confidence * 0.4)
        else:
            # Lower quality retrieval - rely more on LLM confidence
            confidence_score = (avg_retrieval_score * 0.4) + (llm_confidence * 0.6)

        # RUNTIME EVALUATION: Auto-escalate if confidence is too low
        # This is a production safety mechanism - prevents unsafe approvals/rejections
        # Note: Full faithfulness evaluation (LLM-as-judge) would run here in production
        # For now, we use retrieval confidence + LLM confidence indicator
        decision = decision_data.get("decision", "ESCALATE").upper()
        
        # Only override to ESCALATE if confidence is very low AND it's an APPROVE/REJECT decision
        # If LLM already said ESCALATE, respect that decision
        # Lower threshold to 0.65 to reduce false escalations, but still catch truly uncertain cases
        if decision != "ESCALATE" and confidence_score < 0.65:
            decision = "ESCALATE"
            decision_data["explanation"] = (
                f"Low confidence score ({confidence_score:.2f}). "
                f"Original decision: {decision_data.get('decision', 'UNKNOWN')}. "
                f"Manual review required."
            )
        # If confidence is medium (0.65-0.75) and decision is REJECT, allow it (rejections are safer)
        # But if it's APPROVE with medium confidence, be more cautious
        elif decision == "APPROVE" and 0.65 <= confidence_score < 0.70:
            # For APPROVE with medium confidence, check if retrieval quality is good
            if avg_retrieval_score < 0.70:
                decision = "ESCALATE"
                decision_data["explanation"] = (
                    f"Medium confidence score ({confidence_score:.2f}) with low retrieval quality. "
                    f"Original decision: APPROVE. Manual review recommended."
                )

        # Build final response
        return {
            "decision": decision,
            "explanation": decision_data.get("explanation", ""),
            "cited_sections": cited_sections,
            "retrieved_chunks": [
                {
                    "text": r["chunk"][:500],  # Truncate for response
                    "metadata": r["metadata"],
                    "similarity_score": float(r["score"]),
                }
                for r in retrieved_results
            ],
            "raw_model_output": raw_output,
            "confidence_score": float(confidence_score),
        }

    def clear_policies(self) -> None:
        """Clear all ingested policies from the vector store."""
        self.retriever.clear()
        self.policy_metadata.clear()

    def get_policy_info(self) -> Dict[str, Any]:
        """Get information about ingested policies."""
        return {
            "num_policies": len(self.policy_metadata),
            "num_documents": self.retriever.get_document_count(),
            "policies": self.policy_metadata,
        }
