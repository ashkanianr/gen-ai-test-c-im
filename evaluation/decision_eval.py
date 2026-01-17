"""Decision accuracy evaluation module.

This module compares predicted decisions against expected decisions
and calculates accuracy metrics.
"""

from typing import Dict, Any, List
from collections import Counter


def evaluate_decision_accuracy(
    predicted_decision: str,
    expected_decision: str,
) -> Dict[str, Any]:
    """
    Evaluate decision accuracy for a single claim.

    Args:
        predicted_decision: The decision made by the system
        expected_decision: The expected/correct decision

    Returns:
        Dict with is_correct, predicted, expected, and match details
    """
    # Normalize decisions (uppercase, strip)
    predicted = predicted_decision.upper().strip()
    expected = expected_decision.upper().strip()

    is_correct = predicted == expected

    return {
        "is_correct": is_correct,
        "predicted": predicted,
        "expected": expected,
        "match": "CORRECT" if is_correct else "INCORRECT",
    }


def evaluate_batch_decisions(
    predictions: List[Dict[str, str]],
    expected: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Evaluate decision accuracy for a batch of claims.

    Args:
        predictions: List of dicts with 'claim_id' and 'decision' keys
        expected: List of dicts with 'claim_id' and 'decision' keys

    Returns:
        Dict with overall accuracy, per-decision metrics, and confusion matrix
    """
    # Create lookup dicts
    expected_dict = {e["claim_id"]: e["decision"] for e in expected}
    predicted_dict = {p["claim_id"]: p["decision"] for p in predictions}

    # Evaluate each prediction
    results = []
    for claim_id in predicted_dict:
        if claim_id not in expected_dict:
            continue  # Skip if no expected value

        result = evaluate_decision_accuracy(
            predicted_dict[claim_id],
            expected_dict[claim_id],
        )
        result["claim_id"] = claim_id
        results.append(result)

    if not results:
        return {
            "accuracy": 0.0,
            "num_correct": 0,
            "num_total": 0,
            "per_decision_accuracy": {},
            "confusion_matrix": {},
        }

    # Calculate overall accuracy
    num_correct = sum(1 for r in results if r["is_correct"])
    num_total = len(results)
    accuracy = num_correct / num_total if num_total > 0 else 0.0

    # Calculate per-decision accuracy
    decision_groups = {}
    for result in results:
        expected_decision = result["expected"]
        if expected_decision not in decision_groups:
            decision_groups[expected_decision] = []
        decision_groups[expected_decision].append(result)

    per_decision_accuracy = {}
    for decision, group_results in decision_groups.items():
        correct = sum(1 for r in group_results if r["is_correct"])
        total = len(group_results)
        per_decision_accuracy[decision] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "num_correct": correct,
            "num_total": total,
        }

    # Build confusion matrix
    confusion_matrix = {}
    for result in results:
        key = f"{result['predicted']} -> {result['expected']}"
        confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "per_decision_accuracy": per_decision_accuracy,
        "confusion_matrix": confusion_matrix,
        "detailed_results": results,
    }
