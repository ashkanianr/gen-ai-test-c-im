"""Main entry point for Manulife SmartClaim Intelligence.

Supports both CLI and FastAPI server interfaces.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# FastAPI imports
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from app.rag_pipeline import RAGPipeline


# Global pipeline instance (for FastAPI)
pipeline: Optional[RAGPipeline] = None


# Pydantic models for FastAPI
if FASTAPI_AVAILABLE:
    class ClaimRequest(BaseModel):
        claim_text: str
        policy_name: Optional[str] = None

    class PolicyIngestResponse(BaseModel):
        policy_name: str
        num_chunks: int
        status: str

    class ClaimResponse(BaseModel):
        decision: str
        explanation: str
        cited_sections: list
        retrieved_chunks: list
        raw_model_output: str
        confidence_score: float


def create_pipeline() -> RAGPipeline:
    """Create and return a RAG pipeline instance."""
    return RAGPipeline()


def cli_ingest_policy(args):
    """CLI handler for policy ingestion."""
    pipeline = create_pipeline()
    
    try:
        result = pipeline.ingest_policy(
            args.pdf_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"✓ Policy ingested successfully: {result['policy_name']}")
        print(f"  Chunks created: {result['num_chunks']}")
        return 0
    except Exception as e:
        print(f"✗ Error ingesting policy: {e}", file=sys.stderr)
        return 1


def cli_process_claim(args):
    """CLI handler for claim processing."""
    pipeline = create_pipeline()
    
    # Ingest policy if provided
    if args.policy_path:
        try:
            pipeline.ingest_policy(args.policy_path)
            print(f"✓ Policy ingested: {Path(args.policy_path).stem}")
        except Exception as e:
            print(f"✗ Error ingesting policy: {e}", file=sys.stderr)
            return 1
    
    # Process claim
    try:
        # Read claim text if it's a file
        claim_text = args.claim_text
        if Path(claim_text).exists():
            with open(claim_text, "r", encoding="utf-8") as f:
                claim_text = f.read()
        
        result = pipeline.process_claim(claim_text, args.policy_name)
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*60)
            print("CLAIM DECISION")
            print("="*60)
            print(f"Decision: {result['decision']}")
            print(f"Confidence: {result['confidence_score']:.2%}")
            print(f"\nExplanation:\n{result['explanation']}")
            
            if result['cited_sections']:
                print(f"\nCited Sections ({len(result['cited_sections'])}):")
                for i, section in enumerate(result['cited_sections'], 1):
                    print(f"  {i}. {section.get('section_id', 'Unknown')} "
                          f"(Page {section.get('page_number', '?')})")
                    print(f"     {section.get('relevant_text', '')[:200]}...")
        
        return 0
    except Exception as e:
        print(f"✗ Error processing claim: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cli_evaluate(args):
    """CLI handler for evaluation."""
    from evaluation.run_evaluation import run_evaluation
    
    pipeline = create_pipeline()
    
    # Ingest policies from evaluation dataset
    if args.policy_paths:
        for policy_path in args.policy_paths:
            try:
                pipeline.ingest_policy(policy_path)
                print(f"✓ Ingested: {Path(policy_path).stem}")
            except Exception as e:
                print(f"✗ Error ingesting {policy_path}: {e}", file=sys.stderr)
                return 1
    
    # Run evaluation
    try:
        report = run_evaluation(
            pipeline,
            args.dataset_path,
            args.output,
        )
    
        # Print summary
        metrics = report["overall_metrics"]
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"Avg Retrieval Recall: {metrics['avg_retrieval_recall']:.2%}")
        print(f"Avg Faithfulness: {metrics['avg_faithfulness']:.2%}")
        print(f"Avg Composite Confidence: {metrics['avg_composite_confidence']:.2%}")
        print(f"Escalation Rate: {metrics['escalation_rate']:.2%}")
        print(f"Examples Evaluated: {metrics['num_examples']}/{metrics['num_total']}")
        
        if report.get("per_category_metrics"):
            print("\nPer-Category Metrics:")
            for category, cat_metrics in report["per_category_metrics"].items():
                print(f"  {category}: {cat_metrics['accuracy']:.2%} "
                      f"({cat_metrics['count']} examples)")
        
        if args.output:
            print(f"\nFull report saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"✗ Error running evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def setup_fastapi_app() -> FastAPI:
    """Create and configure FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not available. Install with: pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title="Manulife SmartClaim Intelligence API",
        description="AI-powered insurance claim processing system",
        version="1.0.0",
    )
    
    global pipeline
    pipeline = create_pipeline()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "smartclaim-intelligence"}
    
    @app.post("/ingest-policy", response_model=PolicyIngestResponse)
    async def ingest_policy(file: UploadFile = File(...)):
        """
        Upload and ingest a policy PDF.
        
        Args:
            file: Policy PDF file
            
        Returns:
            Policy ingestion result
        """
        try:
            # Save uploaded file temporarily
            temp_path = Path(f"/tmp/{file.filename}")
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Ingest policy
            result = pipeline.ingest_policy(str(temp_path))
            
            # Clean up temp file
            temp_path.unlink()
            
            return PolicyIngestResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/process-claim", response_model=ClaimResponse)
    async def process_claim(request: ClaimRequest):
        """
        Process an insurance claim.
        
        Args:
            request: Claim request with claim_text and optional policy_name
            
        Returns:
            Claim decision with explanation and citations
        """
        try:
            result = pipeline.process_claim(
                request.claim_text,
                request.policy_name,
            )
            return ClaimResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/policies")
    async def list_policies():
        """Get information about ingested policies."""
        try:
            info = pipeline.get_policy_info()
            return info
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manulife SmartClaim Intelligence - AI-powered claim processing"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Policy ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a policy PDF")
    ingest_parser.add_argument("pdf_path", help="Path to policy PDF file")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000,
                              help="Chunk size in characters (default: 1000)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200,
                              help="Chunk overlap in characters (default: 200)")
    
    # Claim processing command
    claim_parser = subparsers.add_parser("claim", help="Process an insurance claim")
    claim_parser.add_argument("claim_text", help="Claim text or path to claim file")
    claim_parser.add_argument("--policy-path", help="Path to policy PDF (if not already ingested)")
    claim_parser.add_argument("--policy-name", help="Policy name filter (if multiple policies)")
    claim_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on dataset")
    eval_parser.add_argument("dataset_path", help="Path to evaluation dataset JSON")
    eval_parser.add_argument("--policy-paths", nargs="+", help="Paths to policy PDFs to ingest")
    eval_parser.add_argument("--output", help="Path to save evaluation report")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "ingest":
        return cli_ingest_policy(args)
    elif args.command == "claim":
        return cli_process_claim(args)
    elif args.command == "evaluate":
        return cli_evaluate(args)
    elif args.command == "serve":
        if not FASTAPI_AVAILABLE:
            print("✗ FastAPI not available. Install with: pip install fastapi uvicorn", file=sys.stderr)
            return 1
        try:
            import uvicorn
            app = setup_fastapi_app()
            uvicorn.run(app, host=args.host, port=args.port)
            return 0
        except Exception as e:
            print(f"✗ Error starting server: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
