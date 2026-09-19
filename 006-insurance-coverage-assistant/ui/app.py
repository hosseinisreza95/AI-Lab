"""Advisor web app.

Deliberately thin: it posts the case to the API and renders what comes back. All
retrieval and grounding logic lives in the backend, so the same answers are
available to the CRM integration without going through this page.

Run with:  streamlit run ui/app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("API_TIMEOUT", "90"))

VERDICT_STYLES = {
    "covered": ("✅", "Covered", "#1b5e20", "#e8f5e9"),
    "covered_with_conditions": ("⚠️", "Covered, with conditions", "#e65100", "#fff3e0"),
    "not_covered": ("❌", "Not covered", "#b71c1c", "#ffebee"),
    "insufficient_information": ("❓", "Not enough information", "#0d47a1", "#e3f2fd"),
    "no_applicable_clause": ("🔍", "No applicable clause found", "#37474f", "#eceff1"),
}

EXAMPLE_CASES = [
    "Customer has been off work since 12 February with a slipped disc confirmed by "
    "MRI. She is on the 2025 income protection. When do the payments start?",

    "Customer is off work with depression, signed off by her GP six weeks ago. She "
    "wants to know if she will be paid and for how long.",

    "Customer wants new glasses. He got a pair reimbursed in March last year and his "
    "prescription has not changed. Health Premium 2024.",

    "Customer's 14-year-old daughter needs braces. Are we covering that?",

    "Customer broke his wrist skiing in Austria and was treated in a clinic there. "
    "Will we reimburse it?",
]

st.set_page_config(page_title="Coverage Assistant", page_icon="📋", layout="wide")


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    try:
        return requests.get(f"{API_URL}/", timeout=10).json()
    except requests.RequestException as exc:
        return {"status": "unreachable", "error": str(exc)}


def render_verdict(payload: dict) -> None:
    icon, label, text_color, background = VERDICT_STYLES.get(
        payload["verdict"], ("•", payload["verdict"], "#263238", "#eceff1")
    )
    st.markdown(
        f"""
        <div style="background:{background};border-left:6px solid {text_color};
                    padding:18px 20px;border-radius:6px;margin:8px 0 18px 0;">
          <div style="color:{text_color};font-size:13px;font-weight:700;
                      letter-spacing:.08em;text-transform:uppercase;">
            {icon} {label}
          </div>
          <div style="color:{text_color};font-size:19px;font-weight:600;margin-top:8px;
                      line-height:1.45;">
            {payload["headline"]}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(payload["explanation"])


def render_sidebar() -> None:
    st.sidebar.title("📋 Coverage Assistant")
    health = fetch_health()

    if health.get("status") != "ok":
        st.sidebar.error(f"API unreachable at {API_URL}")
        st.sidebar.caption(health.get("error", ""))
        return

    if not health.get("index_ready"):
        st.sidebar.warning("Contract index not built.")
        if st.sidebar.button("Build index now"):
            with st.spinner("Indexing contract library..."):
                response = requests.post(f"{API_URL}/ingest", timeout=300)
            st.sidebar.json(response.json())
            fetch_health.clear()
            st.rerun()
        return

    st.sidebar.success("Contract library indexed")
    st.sidebar.subheader("Contracts in scope")
    for contract_id in health.get("contracts", []):
        st.sidebar.write(f"• `{contract_id}`")

    st.sidebar.caption(f"Model: {health.get('chat_model')}")
    st.sidebar.divider()
    st.sidebar.caption(
        "Answers are grounded in the indexed wordings only. Where the case does not "
        "identify a contract generation, the assistant reports the difference "
        "instead of choosing."
    )


def main() -> None:
    render_sidebar()

    st.title("Coverage Assistant")
    st.caption(
        "Describe the customer situation in plain language. The assistant returns "
        "the applicable clauses and what they mean for this case."
    )

    if "case_text" not in st.session_state:
        st.session_state.case_text = ""

    with st.expander("Example cases", expanded=False):
        for index, example in enumerate(EXAMPLE_CASES):
            if st.button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.case_text = example
                st.rerun()

    case = st.text_area(
        "Customer case",
        key="case_text",
        height=120,
        placeholder="e.g. Customer is on sick leave since February with a slipped disc...",
    )

    submitted = st.button("Check coverage", type="primary", disabled=not case.strip())

    if submitted:
        with st.spinner("Searching the contract wordings..."):
            try:
                response = requests.post(
                    f"{API_URL}/coverage", json={"case": case}, timeout=TIMEOUT
                )
                response.raise_for_status()
                st.session_state.result = response.json()
            except requests.HTTPError as exc:
                st.error(f"API error: {exc.response.status_code} — {exc.response.text}")
                return
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                return

    result = st.session_state.get("result")
    if not result:
        return

    st.divider()
    render_verdict(result)

    left, right = st.columns([3, 2])

    with left:
        if result["conditions"]:
            st.subheader("Conditions")
            for condition in result["conditions"]:
                st.write(f"- {condition}")

        if result["what_would_change_the_answer"]:
            st.subheader("What would change this answer")
            for item in result["what_would_change_the_answer"]:
                st.write(f"- {item}")

        if result["advisor_note"]:
            st.info(f"**Advisor note.** {result['advisor_note']}")

    with right:
        st.subheader("Governing clauses")
        for citation in result["governing_clauses"]:
            st.write(f"`{citation}`")

        if result["facts"]:
            st.subheader("Facts read from the case")
            st.json(result["facts"], expanded=False)

        if result["missing_facts"]:
            st.subheader("Not stated in the case")
            for item in result["missing_facts"]:
                st.write(f"- {item}")

    st.divider()
    st.subheader(f"Retrieved clauses ({len(result['clauses'])})")
    st.caption(
        "Every clause the retriever returned, in rank order. Open one to read the "
        "wording the advisor would quote."
    )

    governing = {citation.strip() for citation in result["governing_clauses"]}
    for clause in result["clauses"]:
        short = f"{clause['contract_id']} {clause['article']}"
        used = any(short in citation for citation in governing)
        status_flag = "" if clause["status"] == "active" else f" · {clause['status']}"
        label = f"{'★ ' if used else ''}{short} — {clause['heading']}{status_flag}"

        with st.expander(label, expanded=used):
            st.caption(
                f"{clause['product']} v{clause['version']} · "
                f"fusion score {clause['score']:.5f}"
            )
            st.markdown(clause["text"])
            st.caption("Matched by: " + "; ".join(clause["matched_queries"]))

    with st.expander("Retrieval queries the assistant built", expanded=False):
        st.caption(
            "The advisor's sentence is decomposed into separate clause-level "
            "searches. This is what makes the waiting period and the exclusion both "
            "get found."
        )
        for query in result["search_queries"]:
            st.write(f"- {query}")
        st.caption(f"Answered in {result['latency_ms']} ms by {result['model']}.")


if __name__ == "__main__":
    main()
