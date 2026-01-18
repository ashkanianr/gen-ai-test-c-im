"""Test script to run multiple claims and see results.

This script helps you test different claim scenarios quickly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_pipeline import RAGPipeline


def test_claim(claim_path: str, policy_path: str, expected_decision: str = None):
    """
    Test a single claim and display results.
    
    Args:
        claim_path: Path to claim file
        policy_path: Path to policy file
        expected_decision: Expected decision (APPROVE/REJECT/ESCALATE) for comparison
    """
    print("\n" + "="*80)
    print(f"Testing: {Path(claim_path).name}")
    if expected_decision:
        print(f"Expected Decision: {expected_decision}")
    print("="*80)
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Ingest policy
        print(f"\n[FILE] Ingesting policy: {Path(policy_path).name}")
        policy_result = pipeline.ingest_policy(policy_path)
        print(f"   [OK] Ingested {policy_result['num_chunks']} chunks")
        
        # Process claim
        print(f"\n[CLAIM] Processing claim: {Path(claim_path).name}")
        result = pipeline.process_claim(claim_path)
        
        # Display results
        print(f"\n{'='*80}")
        print("DECISION RESULTS")
        print(f"{'='*80}")
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        
        if expected_decision:
            match = "[OK] CORRECT" if result['decision'] == expected_decision else "[X] MISMATCH"
            print(f"Expected: {expected_decision} - {match}")
        
        print(f"\nExplanation:")
        print(f"{result['explanation'][:500]}...")
        
        if result['cited_sections']:
            print(f"\nCited Sections ({len(result['cited_sections'])}):")
            for i, section in enumerate(result['cited_sections'][:3], 1):  # Show first 3
                print(f"  {i}. {section.get('section_id', 'Unknown')} "
                      f"(Page {section.get('page_number', '?')})")
        
        print(f"\nRetrieved Chunks: {len(result['retrieved_chunks'])}")
        if result['retrieved_chunks']:
            avg_score = sum(c.get('similarity_score', 0) for c in result['retrieved_chunks']) / len(result['retrieved_chunks'])
            print(f"Average Similarity Score: {avg_score:.3f}")
        
        print(f"\n{'='*80}\n")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run test scenarios."""
    
    # Define test cases
    test_cases = [
        # Health Insurance Claims
        {
            "claim": "data/claims/health_claim_001_approve.txt",
            "policy": "data/policies/health_policy.txt",
            "expected": "APPROVE"
        },
        {
            "claim": "data/claims/health_claim_002_approve.txt",
            "policy": "data/policies/health_policy.txt",
            "expected": "APPROVE"
        },
        {
            "claim": "data/claims/health_claim_004_reject.txt",
            "policy": "data/policies/health_policy.txt",
            "expected": "REJECT"
        },
        {
            "claim": "data/claims/health_claim_006_reject.txt",
            "policy": "data/policies/health_policy.txt",
            "expected": "REJECT"
        },
        {
            "claim": "data/claims/health_claim_007_escalate.txt",
            "policy": "data/policies/health_policy.txt",
            "expected": "ESCALATE"
        },
        # Travel Insurance Claims
        {
            "claim": "data/claims/travel_claim_001_approve.txt",
            "policy": "data/policies/travel_policy.txt",
            "expected": "APPROVE"
        },
        {
            "claim": "data/claims/travel_claim_004_reject.txt",
            "policy": "data/policies/travel_policy.txt",
            "expected": "REJECT"
        },
    ]
    
    print("\n" + "="*80)
    print("MANULIFE SMARTCLAIM INTELLIGENCE - TEST SUITE")
    print("="*80)
    print(f"\nRunning {len(test_cases)} test cases...\n")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}]")
        result = test_claim(
            test_case["claim"],
            test_case["policy"],
            test_case["expected"]
        )
        results.append({
            "test": Path(test_case["claim"]).name,
            "expected": test_case["expected"],
            "actual": result["decision"] if result else "ERROR",
            "correct": result and result["decision"] == test_case["expected"]
        })
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    correct = sum(1 for r in results if r["correct"])
    print(f"\nTotal Tests: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Incorrect/Errors: {len(results) - correct}")
    print(f"Accuracy: {correct/len(results)*100:.1f}%")
    
    print("\nDetailed Results:")
    for r in results:
        status = "[OK]" if r["correct"] else "[X]"
        print(f"  {status} {r['test']}: Expected {r['expected']}, Got {r['actual']}")


if __name__ == "__main__":
    main()
