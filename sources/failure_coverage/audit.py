"""Annotate missing deposit outcomes using an independently preserved FDIC register."""

import json
from pathlib import Path
import pandas as pd


def annotate(root):
    folder = root / "sources/failure_coverage"
    register = pd.read_csv(folder / "fdic_failure_register.csv")
    # The complete endpoint includes assistance as well as failures. Keep that distinction.
    register["event_date"] = pd.to_datetime(register.FAILDATE)
    inputs = pd.read_csv(
        root / "growth_outputs/masterclass/evaluation_input_population.csv", parse_dates=["date"]
    )
    unknown = inputs.loc[inputs["Outcome status"].ne("Usable adjacent outcome")].copy()
    records = []
    matches = []
    for row in unknown.itertuples(index=False):
        events = register.loc[register.CERT.eq(row.CERT) & register.event_date.gt(row.date)].copy()
        for event in events.itertuples(index=False):
            matches.append(
                {
                    "CERT": row.CERT,
                    "report_date": row.date,
                    "event_date": event.event_date,
                    "event_type": event.RESTYPE,
                    "event_name": event.NAME,
                    "register_id": event.ID,
                }
            )
        records.append(
            {
                "CERT": row.CERT,
                "NAME": row.NAME,
                "date": row.date,
                "later_failure_records": int(events.RESTYPE.eq("FAILURE").sum()),
                "later_assistance_records": int(events.RESTYPE.eq("ASSISTANCE").sum()),
                "classification": "Verified later register event; balance remains unknown"
                if len(events)
                else "Unresolved; no later register match",
            }
        )
    pd.DataFrame(records).to_csv(folder / "unresolved_outcome_annotations.csv", index=False)
    pd.DataFrame(matches).to_csv(folder / "matched_event_records.csv", index=False)
    counts = register.RESTYPE.value_counts().to_dict()
    (folder / "README.md").write_text(
        "# Unresolved-outcome coverage audit\n\n"
        f"The complete FDIC endpoint contains {len(register):,} records: {counts}. "
        "Assistance and failure records remain distinct. The original response, retrieval metadata, "
        "and fingerprint are preserved. Some certificates occur in more than one historical event; "
        "the event ledger retains each matching later record instead of assuming a unique certificate.\n\n"
        "The audit annotates the 86 unresolved deposit outcomes. An event does not supply an "
        "unobserved balance or establish the cause of its absence. No event labels enter Study 2 "
        "modeling or review-list evaluation. Rebuild annotations with "
        "`python sources/failure_coverage/audit.py`. The latest recorded event is an endpoint "
        "observation, not a guarantee that events through retrieval time are complete.\n"
    )


if __name__ == "__main__":
    annotate(Path(__file__).resolve().parents[2])
