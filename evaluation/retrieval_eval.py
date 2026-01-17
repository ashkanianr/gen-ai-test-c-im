"""Retrieval evaluation module.

This module evaluates the quality of retrieval by checking if required
policy sections appear in the retrieved chunks.
"""

from typing import List, Dict, Any, Set


def calculate_retrieval_recall(
    retrieved_chunks: List[Dict[str, Any]],
    expected_sections: List[str],
) -> Dict[str, Any]:
    """
    Calculate retrieval recall metric.

    Args:
        retrieved_chunks: List of retrieved chunk dicts with metadata
        expected_sections: List of expected section IDs that should be retrieved

    Returns:
        Dict with recall score, missing sections, and found sections
    """
    if not expected_sections:
        return {
            "recall": 1.0,
            "missing_sections": [],
            "found_sections": [],
            "num_expected": 0,
            "num_found": 0,
        }

    # Extract section IDs from retrieved chunks
    retrieved_section_ids = set()
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        section_id = metadata.get("section_id", "")
        if section_id:
            retrieved_section_ids.add(section_id)

    # Normalize section IDs for comparison (case-insensitive, strip whitespace)
    retrieved_normalized = {s.lower().strip() for s in retrieved_section_ids}
    expected_normalized = {s.lower().strip() for s in expected_sections}

    # Calculate recall
    found_sections = expected_normalized.intersection(retrieved_normalized)
    missing_sections = expected_normalized - retrieved_normalized

    recall = len(found_sections) / len(expected_normalized) if expected_normalized else 1.0

    # Map back to original section IDs
    found_original = [
        s for s in expected_sections
        if s.lower().strip() in found_sections
    ]
    missing_original = [
        s for s in expected_sections
        if s.lower().strip() in missing_sections
    ]

    return {
        "recall": recall,
        "missing_sections": missing_original,
        "found_sections": found_original,
        "num_expected": len(expected_sections),
        "num_found": len(found_original),
    }


def evaluate_retrieval_quality(
    retrieved_chunks: List[Dict[str, Any]],
    expected_sections: List[str],
    min_recall_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality with detailed metrics.

    Args:
        retrieved_chunks: List of retrieved chunk dicts
        expected_sections: List of expected section IDs
        min_recall_threshold: Minimum recall to consider retrieval successful

    Returns:
        Dict with evaluation results
    """
    recall_metrics = calculate_retrieval_recall(retrieved_chunks, expected_sections)

    # Check if retrieval meets threshold
    meets_threshold = recall_metrics["recall"] >= min_recall_threshold

    # Calculate average similarity score
    similarity_scores = [
        chunk.get("similarity_score", 0.0)
        for chunk in retrieved_chunks
    ]
    avg_similarity = (
        sum(similarity_scores) / len(similarity_scores)
        if similarity_scores else 0.0
    )

    return {
        **recall_metrics,
        "meets_threshold": meets_threshold,
        "avg_similarity_score": avg_similarity,
        "num_retrieved": len(retrieved_chunks),
        "min_recall_threshold": min_recall_threshold,
    }
