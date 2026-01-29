"""Streamlit UI for Manulife SmartClaim Intelligence.

Ingest policies via sidebar, enter claim text, get decision and explanation in the browser.
Uses the same RAGPipeline as CLI and FastAPI.
"""

import tempfile
from pathlib import Path

import streamlit as st

from app.rag_pipeline import RAGPipeline


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


if __name__ == "__main__":
    main()
