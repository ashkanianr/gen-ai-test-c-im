"""End-to-end evaluation runner.

This module orchestrates all evaluation layers and calculates
composite confidence scores.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from app.llm_client import get_llm_client

from evaluation.retrieval_eval import evaluate_retrieval_quality
from evaluation.faithfulness_eval import evaluate_faithfulness
from evaluation.decision_eval import evaluate_decision_accuracy


def calculate_composite_confidence(
    retrieval_recall: float,
    faithfulness_score: float,
    decision_correct: bool,
    weights: Dict[str, float] = None,
) -> float:
    """
    Calculate composite confidence score.

    Args:
        retrieval_recall: Retrieval recall score (0-1)
        faithfulness_score: Faithfulness score (0-1)
        decision_correct: Whether decision was correct (boolean, converted to 0-1)
        weights: Optional custom weights (default: 40% retrieval, 40% faithfulness, 20% decision)

    Returns:
        Composite confidence score (0-1)
    """
    if weights is None:
        weights = {
            "retrieval": 0.4,
            "faithfulness": 0.4,
            "decision": 0.2,
        }

    decision_score = 1.0 if decision_correct else 0.0

    confidence = (
        weights["retrieval"] * retrieval_recall +
        weights["faithfulness"] * faithfulness_score +
        weights["decision"] * decision_score
    )

    return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]


def evaluate_single_claim(
    pipeline: RAGPipeline,
    claim_text: str,
    expected_decision: str,
    expected_sections: List[str],
    policy_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single claim through the complete pipeline.

    Args:
        pipeline: Initialized RAG pipeline
        claim_text: Claim description
        expected_decision: Expected decision (APPROVE/REJECT/ESCALATE)
        expected_sections: List of expected policy section IDs
        policy_text: Optional full policy text for faithfulness evaluation

    Returns:
        Dict with all evaluation metrics
    """
    # Process claim through pipeline
    result = pipeline.process_claim(claim_text)

    # Extract components
    predicted_decision = result["decision"]
    explanation = result["explanation"]
    retrieved_chunks = result["retrieved_chunks"]
    confidence_score = result["confidence_score"]

    # 1. Retrieval evaluation
    retrieval_eval = evaluate_retrieval_quality(
        retrieved_chunks,
        expected_sections,
    )

    # 2. Faithfulness evaluation
    # Use retrieved chunks as policy text if not provided
    if policy_text is None:
        policy_text = "\n\n".join([
            chunk.get("text", "") for chunk in retrieved_chunks
        ])

    faithfulness_eval = evaluate_faithfulness(
        policy_text,
        explanation,
    )

    # 3. Decision accuracy evaluation
    decision_eval = evaluate_decision_accuracy(
        predicted_decision,
        expected_decision,
    )

    # 4. Calculate composite confidence
    composite_confidence = calculate_composite_confidence(
        retrieval_recall=retrieval_eval["recall"],
        faithfulness_score=faithfulness_eval["faithfulness_score"],
        decision_correct=decision_eval["is_correct"],
    )

    # 5. Determine if should escalate based on confidence
    should_escalate = composite_confidence < 0.75

    return {
        "claim_text": claim_text,
        "predicted_decision": predicted_decision,
        "expected_decision": expected_decision,
        "retrieval_evaluation": retrieval_eval,
        "faithfulness_evaluation": faithfulness_eval,
        "decision_evaluation": decision_eval,
        "composite_confidence": composite_confidence,
        "pipeline_confidence": confidence_score,
        "should_escalate": should_escalate,
        "explanation": explanation,
        "cited_sections": result["cited_sections"],
    }


def run_evaluation(
    pipeline: RAGPipeline,
    evaluation_dataset_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run end-to-end evaluation on a dataset.

    Args:
        pipeline: Initialized RAG pipeline with policies ingested
        evaluation_dataset_path: Path to evaluation dataset JSON
        output_path: Optional path to save evaluation report

    Returns:
        Dict with overall evaluation metrics
    """
    # Load evaluation dataset
    with open(evaluation_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must be a list of examples")

    # Evaluate each example
    results = []
    for i, example in enumerate(dataset):
        claim_text = example["claim_text"]
        expected_decision = example["expected_decision"]
        expected_sections = example.get("expected_sections", [])

        print(f"Evaluating example {i+1}/{len(dataset)}...")

        try:
            result = evaluate_single_claim(
                pipeline,
                claim_text,
                expected_decision,
                expected_sections,
            )
            result["example_id"] = i
            result["claim_category"] = example.get("claim_category", "unknown")
            results.append(result)
        except Exception as e:
            print(f"Error evaluating example {i+1}: {e}")
            results.append({
                "example_id": i,
                "error": str(e),
            })

    # Calculate aggregate metrics
    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        return {
            "error": "No successful evaluations",
            "results": results,
        }

    # Overall accuracy
    correct_decisions = sum(
        1 for r in successful_results
        if r.get("decision_evaluation", {}).get("is_correct", False)
    )
    overall_accuracy = correct_decisions / len(successful_results)

    # Average metrics
    avg_retrieval_recall = sum(
        r["retrieval_evaluation"]["recall"]
        for r in successful_results
    ) / len(successful_results)

    avg_faithfulness = sum(
        r["faithfulness_evaluation"]["faithfulness_score"]
        for r in successful_results
    ) / len(successful_results)

    avg_composite_confidence = sum(
        r["composite_confidence"]
        for r in successful_results
    ) / len(successful_results)

    # Escalation rate
    escalation_count = sum(
        1 for r in successful_results
        if r.get("should_escalate", False) or r["predicted_decision"] == "ESCALATE"
    )
    escalation_rate = escalation_count / len(successful_results)

    # Per-category metrics
    category_metrics = {}
    for category in ["clearly_covered", "clearly_excluded", "ambiguous"]:
        category_results = [
            r for r in successful_results
            if r.get("claim_category") == category
        ]
        if category_results:
            category_correct = sum(
                1 for r in category_results
                if r.get("decision_evaluation", {}).get("is_correct", False)
            )
            category_metrics[category] = {
                "accuracy": category_correct / len(category_results),
                "count": len(category_results),
            }

    evaluation_report = {
        "overall_metrics": {
            "accuracy": overall_accuracy,
            "avg_retrieval_recall": avg_retrieval_recall,
            "avg_faithfulness": avg_faithfulness,
            "avg_composite_confidence": avg_composite_confidence,
            "escalation_rate": escalation_rate,
            "num_examples": len(successful_results),
            "num_total": len(dataset),
        },
        "per_category_metrics": category_metrics,
        "detailed_results": results,
    }

    # Save report if output path provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        print(f"Evaluation report saved to: {output_path}")

    return evaluation_report
