"""
config.py
---------
Central configuration for Smart Inbox Assistant.

Set LLM_PROVIDER to "openai" or "gemini" to control which API the pipeline
calls. All provider-specific settings (model name, API key env var) live
here so the rest of the codebase never has to branch on provider logic.

API keys are read from environment variables — never hardcode credentials.
Copy `.env.example` to `.env` and fill in your key(s), or export the
variables directly in your shell / Streamlit Cloud secrets.
"""

import os
from dataclasses import dataclass

# Load a local .env file if python-dotenv is installed (optional convenience,
# not required in production / Streamlit Cloud where secrets are injected
# as real environment variables).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# On Streamlit Community Cloud, secrets are configured in the app's
# "Secrets" panel (TOML format) and surfaced via st.secrets rather than
# real OS environment variables. Mirror them into os.environ so the same
# os.getenv() calls work identically locally and when deployed.
try:
    import streamlit as st
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass


# ---------------------------------------------------------------------------
# PROVIDER TOGGLE — this is the single switch that controls the whole app
# ---------------------------------------------------------------------------
# Options: "openai" | "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    model: str


PROVIDER_SETTINGS = {
    "openai": ProviderConfig(
        name="openai",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    ),
    "gemini": ProviderConfig(
        name="gemini",
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    ),
}


def get_active_provider() -> ProviderConfig:
    """Return the ProviderConfig for whichever provider is currently active."""
    if LLM_PROVIDER not in PROVIDER_SETTINGS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Must be 'openai' or 'gemini'."
        )
    return PROVIDER_SETTINGS[LLM_PROVIDER]


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("SIA_DATA_PATH", "data/sample_emails.json")
OUTPUT_JSON_PATH = os.getenv("SIA_OUTPUT_JSON", "outputs/processed_emails.json")
OUTPUT_CSV_PATH = os.getenv("SIA_OUTPUT_CSV", "outputs/processed_emails.csv")

# Classification taxonomy — kept centralized so prompts and downstream
# validation always agree on valid labels.
INTENT_LABELS = ["complaint", "inquiry", "order_status", "spam"]
URGENCY_LABELS = ["low", "medium", "high"]
