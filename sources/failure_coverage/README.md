# Unresolved-outcome coverage audit

The complete FDIC endpoint contains 4,117 records: {'FAILURE': 3524, 'ASSISTANCE': 593}. Assistance and failure records remain distinct. The original response, retrieval metadata, and fingerprint are preserved. Some certificates occur in more than one historical event; the event ledger retains each matching later record instead of assuming a unique certificate.

The audit annotates the 86 unresolved deposit outcomes. An event does not supply an unobserved balance or establish the cause of its absence. No event labels enter Study 2 modeling or review-list evaluation. Rebuild annotations with `python sources/failure_coverage/audit.py`. The latest recorded event is an endpoint observation, not a guarantee that events through retrieval time are complete.
