"""Streamlit UI for Manulife SmartClaim Intelligence.

Ingest policies via sidebar, enter claim text, get decision and explanation in the browser.
Uses the same RAGPipeline as CLI and FastAPI.
"""

import tempfile
import time
from pathlib import Path

import streamlit as st

from app.rag_pipeline import RAGPipeline

# Project root for resolving data paths (same as scripts/test_claims.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same 8 test cases as scripts/test_claims.py
TEST_CASES = [
    {"claim": "data/claims/health_claim_001_approve.txt", "policy": "data/policies/health_policy.txt", "expected": "APPROVE"},
    {"claim": "data/claims/health_claim_002_approve.txt", "policy": "data/policies/health_policy.txt", "expected": "APPROVE"},
    {"claim": "data/claims/health_claim_004_reject.txt", "policy": "data/policies/health_policy.txt", "expected": "REJECT"},
    {"claim": "data/claims/health_claim_006_reject.txt", "policy": "data/policies/health_policy.txt", "expected": "REJECT"},
    {"claim": "data/claims/health_claim_007_escalate.txt", "policy": "data/policies/health_policy.txt", "expected": "ESCALATE"},
    {"claim": "data/claims/travel_claim_001_approve.txt", "policy": "data/policies/travel_policy.txt", "expected": "APPROVE"},
    {"claim": "data/claims/travel_claim_004_reject.txt", "policy": "data/policies/travel_policy.txt", "expected": "REJECT"},
]


def get_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline in session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


def main() -> None:
    st.set_page_config(
        page_title="SmartClaim Intelligence",
        page_icon="📋",
        layout="wide",
    )
    st.title("Manulife SmartClaim Intelligence")
    st.caption("Policy-grounded claim adjudication with RAG")

    pipeline = get_pipeline()

    # --- Sidebar: Policy ingestion ---
    with st.sidebar:
        st.header("Policy ingestion")
        st.markdown("Upload a policy PDF or load a sample policy.")

        # File uploader for PDF
        uploaded_file = st.file_uploader(
            "Upload policy PDF",
            type=["pdf"],
            help="Upload a policy document (PDF).",
        )
        if uploaded_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                result = pipeline.ingest_policy(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)
                st.success(f"Ingested: **{result['policy_name']}** ({result['num_chunks']} chunks)")
            except Exception as e:
                st.error(f"Failed to ingest policy: {e}")

        # Optional: load from data/policies if present
        data_policies = Path("data/policies")
        if data_policies.exists():
            policy_files = sorted(data_policies.glob("*.txt")) + sorted(data_policies.glob("*.pdf"))
            if policy_files:
                st.markdown("---")
                st.markdown("Or load a sample policy:")
                sample_name = st.selectbox(
                    "Sample policy",
                    options=[""] + [f.name for f in policy_files],
                    format_func=lambda x: "(select)" if x == "" else x,
                )
                if sample_name and st.button("Load sample policy", key="load_sample"):
                    sample_path = data_policies / sample_name
                    try:
                        result = pipeline.ingest_policy(str(sample_path))
                        st.success(f"Ingested: **{result['policy_name']}** ({result['num_chunks']} chunks)")
                    except Exception as e:
                        st.error(f"Failed to load sample: {e}")

        # List ingested policies
        st.markdown("---")
        info = pipeline.get_policy_info()
        if info["num_policies"] > 0:
            st.markdown("**Ingested policies**")
            for name, meta in info.get("policies", {}).items():
                st.markdown(f"- {name} ({meta.get('num_chunks', '?')} chunks)")
        else:
            st.info("No policies ingested yet. Upload a PDF or load a sample above.")

    # --- Main area: Claim processing ---
    st.header("Process claim")

    # Optional claim file upload (if provided, use as default text)
    claim_file = st.file_uploader(
        "Or upload claim from file",
        type=["txt"],
        help="Upload a .txt file containing the claim.",
    )
    default_claim = ""
    if claim_file is not None:
        default_claim = claim_file.read().decode("utf-8")

    claim_text = st.text_area(
        "Claim text",
        value=default_claim,
        height=150,
        placeholder="Paste or type the claim description here..." if not default_claim else "",
        help="The claim to evaluate against the ingested policy.",
    )

    # Policy name filter if multiple
    info = pipeline.get_policy_info()
    policy_names = list(info.get("policies", {}).keys())
    policy_name = None
    if len(policy_names) > 1:
        policy_name = st.selectbox(
            "Policy to use",
            options=policy_names,
            help="Select which ingested policy to use for this claim.",
        )
    elif len(policy_names) == 1:
        policy_name = policy_names[0]

    process_clicked = st.button("Process claim")

    if process_clicked:
        if not claim_text or not claim_text.strip():
            st.warning("Please enter claim text.")
        elif info["num_policies"] == 0:
            st.error("No policy ingested. Ingest a policy in the sidebar first.")
        else:
            with st.spinner("Processing claim..."):
                try:
                    result = pipeline.process_claim(claim_text.strip(), policy_name)

                    # Decision with styling
                    decision = result["decision"]
                    if decision == "APPROVE":
                        st.success(f"**Decision: APPROVE**")
                    elif decision == "REJECT":
                        st.error("**Decision: REJECT**")
                    else:
                        st.warning("**Decision: ESCALATE**")

                    st.metric("Confidence", f"{result['confidence_score']:.0%}")

                    st.subheader("Explanation")
                    st.write(result["explanation"])

                    if result.get("cited_sections"):
                        with st.expander("Cited sections"):
                            for i, section in enumerate(result["cited_sections"], 1):
                                sid = section.get("section_id", "Unknown")
                                page = section.get("page_number", "?")
                                text = section.get("relevant_text", "")[:300]
                                st.markdown(f"**{i}. {sid}** (Page {page})")
                                st.caption(text)
                                st.markdown("---")

                    with st.expander("Retrieved chunks"):
                        for i, chunk in enumerate(result.get("retrieved_chunks", [])[:5], 1):
                            meta = chunk.get("metadata", {})
                            score = chunk.get("similarity_score", 0)
                            st.markdown(f"**Chunk {i}** (score: {score:.3f}) — {meta.get('section_id', '')} p.{meta.get('page_number', '?')}")
                            st.caption(chunk.get("text", "")[:200] + "…" if len(chunk.get("text", "")) > 200 else chunk.get("text", ""))
                            st.markdown("---")

                    with st.expander("Raw model output"):
                        st.code(result.get("raw_model_output", ""), language="json")

                except Exception as e:
                    st.error(f"Error processing claim: {e}")

    # --- Run test suite (same as scripts/test_claims.py) ---
    st.markdown("---")
    st.subheader("Run test suite")
    st.caption("Run the same 8 sample claims as scripts/test_claims.py. A short delay between tests helps avoid API rate limits.")
    run_suite_clicked = st.button("Run test suite", key="run_test_suite")

    if run_suite_clicked:
        results = []
        progress_placeholder = st.empty()
        summary_placeholder = st.empty()
        table_placeholder = st.empty()
        with st.spinner("Running test suite..."):
            for i, tc in enumerate(TEST_CASES):
                progress_placeholder.progress((i + 1) / len(TEST_CASES), text=f"Test {i + 1}/{len(TEST_CASES)}: {Path(tc['claim']).name}")
                claim_path = PROJECT_ROOT / tc["claim"]
                policy_path = PROJECT_ROOT / tc["policy"]
                if not claim_path.exists() or not policy_path.exists():
                    results.append({
                        "claim": Path(tc["claim"]).name,
                        "expected": tc["expected"],
                        "actual": "ERROR",
                        "correct": False,
                        "error": f"Missing file: {claim_path} or {policy_path}",
                    })
                    continue
                try:
                    pipeline.retriever.clear()
                    pipeline.ingest_policy(str(policy_path))
                    result = pipeline.process_claim(str(claim_path))
                    actual = result["decision"]
                    results.append({
                        "claim": Path(tc["claim"]).name,
                        "expected": tc["expected"],
                        "actual": actual,
                        "correct": actual == tc["expected"],
                        "confidence": result.get("confidence_score"),
                    })
                except Exception as e:
                    results.append({
                        "claim": Path(tc["claim"]).name,
                        "expected": tc["expected"],
                        "actual": "ERROR",
                        "correct": False,
                        "error": str(e),
                    })
                if i < len(TEST_CASES) - 1:
                    time.sleep(2)
        progress_placeholder.empty()
        correct = sum(1 for r in results if r["correct"])
        summary_placeholder.metric("Correct", f"{correct} / {len(TEST_CASES)}")
        summary_placeholder.caption(f"Accuracy: {correct / len(TEST_CASES) * 100:.1f}%")
        rows = []
        for r in results:
            status = "OK" if r["correct"] else "MISMATCH" if r["actual"] != "ERROR" else "ERROR"
            rows.append({
                "Claim": r["claim"],
                "Expected": r["expected"],
                "Actual": r["actual"],
                "Status": status,
                "Confidence": f"{r.get('confidence', 0):.0%}" if r.get("confidence") is not None else "",
            })
        table_placeholder.dataframe(rows, width="stretch", hide_index=True)
        for r in results:
            if r.get("error"):
                with st.expander(f"Error: {r['claim']}"):
                    st.code(r["error"])


if __name__ == "__main__":
    main()
