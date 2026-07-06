"""
dashboard.py
------------
Streamlit application for Smart Inbox Assistant.

Lets a reviewer browse the processed email batch: filter by intent/urgency,
inspect extracted structured data, read the AI-drafted reply, and see
aggregate volume charts. Can also trigger a fresh batch run against the
active LLM provider directly from the sidebar.

Run with:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from processor import load_emails, process_batch, save_outputs

st.set_page_config(
    page_title="Smart Inbox Assistant",
    page_icon="\U0001F4E9",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Minimal custom styling — slate/steel palette distinct from Streamlit defaults
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0F1720; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #E8EDF2 !important; }
    p, li, span, label, .stMarkdown { color: #C4CDD6; }
    div[data-testid="stMetric"] {
        background-color: #182531;
        border: 1px solid #26333F;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #8FA1B3 !important; }
    div[data-testid="stMetricValue"] { color: #F2A65A !important; }
    .urgency-high { color: #E5484D; font-weight: 700; }
    .urgency-medium { color: #F2A65A; font-weight: 600; }
    .urgency-low { color: #4CAF7D; font-weight: 600; }
    .reply-box {
        background-color: #182531;
        border-left: 3px solid #F2A65A;
        border-radius: 6px;
        padding: 12px 16px;
        color: #E8EDF2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

URGENCY_CLASS = {"high": "urgency-high", "medium": "urgency-medium", "low": "urgency-low"}
INTENT_ICON = {
    "complaint": "\U0001F6A8",
    "inquiry": "\u2753",
    "order_status": "\U0001F4E6",
    "spam": "\U0001F6AB",
}


@st.cache_data
def load_results(json_path: str) -> pd.DataFrame:
    p = Path(json_path)
    if not p.exists():
        return pd.DataFrame()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def run_pipeline_now(limit: int | None):
    emails = load_emails(config.DATA_PATH)
    if limit:
        emails = emails[:limit]
    with st.spinner(f"Processing {len(emails)} emails via {config.get_active_provider().name}..."):
        results = process_batch(emails, verbose=False)
        save_outputs(results, config.OUTPUT_JSON_PATH, config.OUTPUT_CSV_PATH)
    st.cache_data.clear()
    return results


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline Controls")
    provider = config.get_active_provider()
    st.markdown(f"**Active provider:** `{provider.name}`  \n**Model:** `{provider.model}`")

    if not provider.api_key:
        st.warning(
            "No API key detected for this provider. The pipeline will run in "
            "**mock mode** (deterministic local classifier) so the Application "
            "stays fully functional without credentials.",
            icon="⚠️",
        )

    st.divider()
    batch_limit = st.slider("Emails to process", min_value=1, max_value=12, value=12)
    if st.button("Run pipeline on sample inbox", type="primary", use_container_width=True):
        run_pipeline_now(batch_limit)
        st.success("Batch complete — results refreshed below.")

    st.divider()
    st.caption(
        "Switch providers by setting `LLM_PROVIDER=openai` or `LLM_PROVIDER=gemini` "
        "as an environment variable (or in Streamlit Cloud → Settings → Secrets), "
        "along with the matching API key."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📩 Smart Inbox Assistant")
st.caption(
    "AI-powered triage for customer support inboxes — classify intent, extract "
    "structured data, and draft replies automatically."
)

df = load_results(config.OUTPUT_JSON_PATH)

if df.empty:
    st.info(
        "No processed results yet. Click **'Run pipeline on sample inbox'** in the "
        "sidebar to classify the sample dataset and populate this dashboard."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
total = len(df)
complaint_ct = int((df["intent"] == "complaint").sum())
order_ct = int((df["intent"] == "order_status").sum())
spam_ct = int((df["intent"] == "spam").sum())
high_urgency_ct = int((df["urgency"] == "high").sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Emails", total)
m2.metric("Complaints", complaint_ct)
m3.metric("Order Status", order_ct)
m4.metric("Spam Filtered", spam_ct)
m5.metric("High Urgency", high_urgency_ct)

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Volume by Intent")
    st.bar_chart(df["intent"].value_counts())
with c2:
    st.subheader("Volume by Urgency")
    urgency_order = ["high", "medium", "low"]
    urgency_counts = df["urgency"].value_counts().reindex(urgency_order).fillna(0)
    st.bar_chart(urgency_counts)

st.divider()

# ---------------------------------------------------------------------------
# Filters + email table/detail view
# ---------------------------------------------------------------------------
st.subheader("Inbox")

f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    intent_filter = st.multiselect(
        "Filter by intent", options=sorted(df["intent"].dropna().unique()), default=[]
    )
with f2:
    urgency_filter = st.multiselect(
        "Filter by urgency", options=["high", "medium", "low"], default=[]
    )
with f3:
    search = st.text_input("Search subject or customer name", "")

filtered = df.copy()
if intent_filter:
    filtered = filtered[filtered["intent"].isin(intent_filter)]
if urgency_filter:
    filtered = filtered[filtered["urgency"].isin(urgency_filter)]
if search:
    mask = (
        filtered["subject"].str.contains(search, case=False, na=False)
        | filtered["customer_name"].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

st.caption(f"Showing {len(filtered)} of {total} emails")

for _, row in filtered.iterrows():
    icon = INTENT_ICON.get(row["intent"], "📧")
    urgency_html = f'<span class="{URGENCY_CLASS.get(row["urgency"], "")}">{row["urgency"].upper()}</span>'
    with st.expander(f"{icon}  {row['subject']}  —  {row['customer_name']}"):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**From:** {row['from_name']} ({row['from_email']})")
            st.markdown(f"**Intent:** `{row['intent']}`")
            st.markdown(f"**Urgency:** {urgency_html}", unsafe_allow_html=True)
            st.markdown(f"**Issue type:** {row['issue_type']}")
            st.markdown(f"**Order #:** {row['order_number'] or '—'}")
        with col_b:
            st.markdown(f"**Processed via:** `{row['processing_mode']}`")
            st.markdown(f"**Time:** {row['processing_seconds']}s")

        if row["suggested_reply"]:
            st.markdown("**Suggested Reply Draft:**")
            st.markdown(f'<div class="reply-box">{row["suggested_reply"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown("*No reply drafted — flagged as spam.*")

st.divider()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.subheader("Export")
e1, e2 = st.columns(2)
with e1:
    st.download_button(
        "Download results as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="processed_emails.csv",
        mime="text/csv",
        use_container_width=True,
    )
with e2:
    st.download_button(
        "Download results as JSON",
        data=json.dumps(filtered.to_dict(orient="records"), indent=2).encode("utf-8"),
        file_name="processed_emails.json",
        mime="application/json",
        use_container_width=True,
    )
