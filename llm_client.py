"""
llm_client.py
-------------
Thin abstraction layer over the OpenAI and Google Gemini chat completion
APIs. The rest of the application calls `get_structured_response(prompt)`
and never needs to know which provider is active — that's decided once,
centrally, in config.py.

Both providers are prompted to return strict JSON so the output is
directly parseable into the pipeline's structured fields (intent, urgency,
extracted entities, suggested reply).

MOCK MODE:
If no API key is configured for the active provider, the client
automatically falls back to a deterministic local mock responder. This
keeps the project fully runnable end-to-end (including the Streamlit
dashboard) for anyone reviewing the code without needing to provision
API credentials first. Mock mode is loud about the fact that it's active
— it is never silently used in place of a real call.
"""

import json
import re
from config import get_active_provider


SYSTEM_PROMPT = """You are an AI assistant for a customer support inbox. \
For the email you are given, respond with STRICT JSON ONLY (no markdown \
fences, no commentary) matching exactly this schema:

{
  "intent": "complaint" | "inquiry" | "order_status" | "spam",
  "urgency": "low" | "medium" | "high",
  "customer_name": string,
  "issue_type": string,
  "order_number": string or null,
  "suggested_reply": string
}

Guidance:
- "complaint": customer is unhappy about a product, service, or experience.
- "inquiry": a general question (policy, product info, shipping, etc.).
- "order_status": customer is asking where an order is / its status.
- "spam": unsolicited, phishing, prize scams, or irrelevant marketing.
- urgency "high": angry tone, financial harm, time-sensitive event, repeated issue.
- urgency "medium": real issue but no immediate deadline or escalation signal.
- urgency "low": casual question, no problem to solve.
- suggested_reply: a warm, professional 2-4 sentence draft reply a human \
agent could send with minimal editing. For spam, suggested_reply should be \
an empty string.
- order_number: extract if present in the email text, otherwise null.
Return ONLY the JSON object.
"""


class LLMError(Exception):
    pass


def _build_user_prompt(email: dict) -> str:
    return (
        f"From: {email.get('from_name', 'unknown')} <{email.get('from_email', '')}>\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Body:\n{email.get('body', '')}"
    )


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```json fences despite instructions;
    strip those defensively before parsing."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Provider-specific calls
# ---------------------------------------------------------------------------

def _call_openai(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    )
    response = gemini_model.generate_content(user_prompt)
    return response.text


# ---------------------------------------------------------------------------
# Mock mode — deterministic, no network/API key required
# ---------------------------------------------------------------------------

def _contains_any(text: str, phrases: list[str]) -> bool:
    """Whole-word/phrase containment check — avoids false positives like
    'won' matching inside 'wondering'."""
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _mock_response(email: dict) -> dict:
    body = (email.get("body") or "").lower()
    subject = (email.get("subject") or "").lower()
    text = f"{subject} {body}"

    order_match = re.search(r"#([A-Za-z]?\d{4,6})", email.get("body", "") or "")
    order_number = order_match.group(1) if order_match else None

    complaint_signals = ["damaged", "wrong item", "shattered", "charged twice", "unacceptable",
                          "frustrated", "disappointing", "upset", "refund", "considering just"]

    if _contains_any(text, ["you have won", "claim now", "click the link", "suspended",
                            "verify now", "prize", "claim your winnings"]):
        intent, urgency = "spam", "low"
    elif _contains_any(text, complaint_signals):
        intent, urgency = "complaint", "high"
    elif _contains_any(text, ["where is my order", "order status", "status of order", "shipped yet",
                              "tracking", "hasn't shown up", "hasn't arrived", "never arrived",
                              "estimated delivery", "shipping confirmation"]):
        intent, urgency = "order_status", "medium"
    elif _contains_any(text, ["question", "wondering", "curious", "do you ship", "return policy",
                              "sizing", "chart", "replacement blades", "out of curiosity"]):
        intent, urgency = "inquiry", "low"
    else:
        intent, urgency = "inquiry", "medium"

    if intent == "spam":
        issue_type = "phishing_or_scam"
        reply = ""
    elif intent == "complaint":
        issue_type = "product_or_order_issue"
        reply = (
            f"Hi {email.get('from_name', 'there').split()[0]}, I'm really sorry for the trouble this has "
            "caused. I'm escalating this right now and will make sure it's resolved quickly — "
            "you'll hear back from us with next steps shortly."
        )
    elif intent == "order_status":
        issue_type = "shipping_status_check"
        reply = (
            f"Hi {email.get('from_name', 'there').split()[0]}, thanks for checking in — let me pull up "
            "the latest tracking details on your order and get you a status update right away."
        )
    else:
        issue_type = "general_question"
        reply = (
            f"Hi {email.get('from_name', 'there').split()[0]}, thanks for reaching out! Happy to help "
            "answer that — here are the details you're looking for."
        )

    return {
        "intent": intent,
        "urgency": urgency,
        "customer_name": email.get("from_name", "Unknown"),
        "issue_type": issue_type,
        "order_number": order_number,
        "suggested_reply": reply,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _validate_api_key(key: str, provider_name: str) -> str:
    """Catch corrupted secrets (smart-quote/dash autocorrect, stray
    whitespace, copy-paste artifacts) before they hit the HTTP layer as an
    opaque UnicodeEncodeError. Fails loudly with a specific, actionable
    message instead of silently degrading to mock mode."""
    cleaned = key.strip()

    try:
        cleaned.encode("ascii")
    except UnicodeEncodeError as exc:
        bad_char = exc.object[exc.start:exc.end]
        raise LLMError(
            f"{provider_name} API key contains a non-ASCII character "
            f"({bad_char!r} at position {exc.start}). This almost always "
            f"means a straight hyphen/quote was autocorrected into a "
            f"'smart' character during copy-paste (e.g. via Word, Notes, "
            f"or Google Docs). Fix: regenerate the key and paste it "
            f"directly from a plain-text source into Streamlit Secrets."
        )

    if cleaned != key:
        raise LLMError(
            f"{provider_name} API key has leading/trailing whitespace or "
            f"a hidden character. Re-paste it without extra spaces/newlines."
        )

    return cleaned


def get_structured_response(email: dict) -> dict:
    """Classify + extract + draft a reply for a single email dict.

    Returns a dict matching the schema described in SYSTEM_PROMPT, plus a
    "mode" key ("openai", "gemini", or "mock") so callers/UI can be
    transparent about whether a live API or the local mock responder
    produced the result.
    """
    provider = get_active_provider()
    user_prompt = _build_user_prompt(email)

    if not provider.api_key:
        result = _mock_response(email)
        result["mode"] = "mock"
        return result

    try:
        validated_key = _validate_api_key(provider.api_key, provider.name)
        if provider.name == "openai":
            raw = _call_openai(SYSTEM_PROMPT, user_prompt, provider.model, validated_key)
        elif provider.name == "gemini":
            raw = _call_gemini(SYSTEM_PROMPT, user_prompt, provider.model, validated_key)
        else:
            raise LLMError(f"Unsupported provider: {provider.name}")

        result = _extract_json(raw)
        result["mode"] = provider.name
        return result

    except Exception as exc:
        # Fail safe rather than crashing the whole batch on one bad email.
        result = _mock_response(email)
        result["mode"] = f"mock_fallback_after_error: {exc}"
        return result
