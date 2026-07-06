"""
processor.py
------------
Core batch pipeline for Smart Inbox Assistant.

Reads a batch of customer emails (JSON or CSV), runs each one through the
active LLM provider (see config.py) to classify intent, extract structured
data, and draft a suggested reply, then writes the enriched results to
both JSON and CSV in outputs/.

Usage:
    python processor.py
    python processor.py --input data/sample_emails.json --limit 5
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config
from llm_client import get_structured_response


def load_emails(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p).to_dict(orient="records")
    else:
        raise ValueError("Input file must be .json or .csv")


def process_batch(emails: list[dict], verbose: bool = True) -> list[dict]:
    provider = config.get_active_provider()
    results = []

    for i, email in enumerate(emails, start=1):
        if verbose:
            print(f"[{i}/{len(emails)}] Processing {email.get('email_id', '?')}: "
                  f"'{email.get('subject', '')[:60]}'")

        start = time.time()
        structured = get_structured_response(email)
        elapsed = time.time() - start

        record = {
            "email_id": email.get("email_id"),
            "from_name": email.get("from_name"),
            "from_email": email.get("from_email"),
            "subject": email.get("subject"),
            "intent": structured.get("intent"),
            "urgency": structured.get("urgency"),
            "customer_name": structured.get("customer_name"),
            "issue_type": structured.get("issue_type"),
            "order_number": structured.get("order_number"),
            "suggested_reply": structured.get("suggested_reply"),
            "processing_mode": structured.get("mode"),
            "processing_seconds": round(elapsed, 3),
        }
        results.append(record)

        if verbose:
            print(f"    -> intent={record['intent']}, urgency={record['urgency']}, "
                  f"mode={record['processing_mode']}")

    return results


def save_outputs(results: list[dict], json_path: str, csv_path: str) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Smart Inbox Assistant batch processor")
    parser.add_argument("--input", default=config.DATA_PATH, help="Path to input JSON/CSV")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N emails")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-email progress logs")
    args = parser.parse_args()

    provider = config.get_active_provider()
    print(f"Active provider: {provider.name} (model: {provider.model})")
    if not provider.api_key:
        print("No API key found for this provider — running in MOCK MODE. "
              "Set the relevant environment variable to call a live LLM.\n")

    emails = load_emails(args.input)
    if args.limit:
        emails = emails[: args.limit]

    print(f"Loaded {len(emails)} emails from {args.input}\n")

    results = process_batch(emails, verbose=not args.quiet)
    save_outputs(results, config.OUTPUT_JSON_PATH, config.OUTPUT_CSV_PATH)

    print(f"\nDone. Wrote {len(results)} records to:")
    print(f"  - {config.OUTPUT_JSON_PATH}")
    print(f"  - {config.OUTPUT_CSV_PATH}")

    # Quick summary
    df = pd.DataFrame(results)
    if not df.empty:
        print("\nIntent breakdown:")
        print(df["intent"].value_counts().to_string())
        print("\nUrgency breakdown:")
        print(df["urgency"].value_counts().to_string())


if __name__ == "__main__":
    main()
