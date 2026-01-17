"""Faithfulness evaluation module.

This module uses LLM-as-judge to verify that every statement in the
decision explanation is supported by the policy text.
"""

from typing import Dict, Any, List
import json
from pathlib import Path

from app.llm_client import get_llm_client, LLMClient


def evaluate_faithfulness(
    policy_text: str,
    explanation: str,
    llm_client: LLMClient = None,
) -> Dict[str, Any]:
    """
    Evaluate faithfulness of decision explanation using LLM-as-judge.

    Args:
        policy_text: The policy text that should support the explanation
        explanation: The decision explanation to evaluate
        llm_client: LLM client instance (auto-created if None)

    Returns:
        Dict with faithfulness_score, unsupported_claims, and justification
    """
    if llm_client is None:
        llm_client = get_llm_client()

    # Load faithfulness prompt template
    prompt_path = Path(__file__).parent.parent / "app" / "prompts" / "judge_faithfulness.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Format prompt
    formatted_prompt = prompt_template.format(
        policy_text=policy_text,
        explanation=explanation,
    )

    # Generate evaluation
    messages = [
        {"role": "user", "content": formatted_prompt}
    ]

    try:
        response = llm_client.chat_completion(
            messages=messages,
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=1500,
        )
        raw_output = response["content"]
    except Exception as e:
        # Fallback: return low faithfulness score
        return {
            "faithfulness_score": 0.0,
            "unsupported_claims": [f"Error evaluating faithfulness: {str(e)}"],
            "supported_claims": [],
            "justification": "Failed to evaluate faithfulness due to LLM error.",
        }

    # Parse LLM response (expecting JSON)
    try:
        # Try to extract JSON from response
        json_text = raw_output
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        evaluation_data = json.loads(json_text)
    except (json.JSONDecodeError, KeyError):
        # Fallback: try to extract score from text
        faithfulness_score = 0.5  # Default to medium if parsing fails
        unsupported_claims = []
        supported_claims = []

        # Try to infer score from text
        if "high" in raw_output.lower() or "1.0" in raw_output or "0.9" in raw_output:
            faithfulness_score = 0.9
        elif "low" in raw_output.lower() or "0.0" in raw_output or "0.1" in raw_output:
            faithfulness_score = 0.2

        evaluation_data = {
            "faithfulness_score": faithfulness_score,
            "unsupported_claims": unsupported_claims,
            "supported_claims": supported_claims,
            "justification": raw_output,
        }

    # Ensure score is in valid range
    faithfulness_score = float(evaluation_data.get("faithfulness_score", 0.5))
    faithfulness_score = max(0.0, min(1.0, faithfulness_score))

    return {
        "faithfulness_score": faithfulness_score,
        "unsupported_claims": evaluation_data.get("unsupported_claims", []),
        "supported_claims": evaluation_data.get("supported_claims", []),
        "justification": evaluation_data.get("justification", ""),
    }
