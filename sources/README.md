# Source review and data decisions

Reviewed September 21, 2026. Monetary FDIC balances in the supplied extract use thousands of USD.

- [Federal Reserve, February 17, 2026](https://www.federalreserve.gov/econres/notes/feds-notes/assessing-bank-resilience-to-a-funding-shock-20260217.html): deposits represent approximately two-thirds of bank liabilities; replacement wholesale funding can cost more.
- [FDIC, May 14, 2026](https://www.fdic.gov/news/press-releases/2026/fdic-releases-staff-study-deposit-flows-three-failed-banks-spring-2023): transaction-level, day-by-day evidence at three failed banks. This motivates the question but does not validate quarterly run detection.
- [FDIC API field dictionary](https://api.fdic.gov/banks/docs/risview_properties.yaml): saved as `fdic_fields.yaml`. DEPDOM is domestic deposits; ASSET total assets; CHBAL cash and due from depository institutions; LNLSNET net loans and leases; EQ equity capital. DEPUNA and DEPUNINS expose related uninsured-deposit items; CALLFORM and BKCLASS describe reporting context.
- [FDIC 2012 reporting transition](https://www.fdic.gov/news/inactive-financial-institution-letters/2012/fil12010.html): savings associations converted from Thrift Financial Reports to Call Reports effective March 31, 2012. Local data contain 5,738 form-100 observations in 2010–2011. No verified historical five-field crosswalk has been established here, so the 2010–2024 comparability gate is not passed. This does not prove that all older data are unusable.
- [FDIC reporting methodology](https://banks.data.fdic.gov/bankfind-suite/help?helpTopic=disclaimers-and-methodologies): BankFind incorporates Call Reports and pre-2012 Thrift Financial Reports; reporting differences and source notes govern comparisons.
- [FDIC uninsured-deposit guidance](https://www.fdic.gov/news/financial-institution-letters/2023/estimated-uninsured-deposits-reporting-expectations) and [2024 RC-O instructions](https://www.fdic.gov/system/files/2024-05/031-041-324-rc-o.pdf): the reporting requirement applies to institutions meeting the USD 1 billion asset threshold. Current-quarter asset grouping is descriptive; it does not reproduce historical eligibility rules.

## Audits and boundaries

The two financial extracts have identical 2020–2024 overlap. The long extract has 359,497 rows, 60 quarters, and no duplicate certificate/report-date keys. Its 1,018 missing equity observations are all on CALLFORM 2 (478 class NC; 540 class OI). The new core uses insured domestic classes N, NM, SM, SB, SI, SL and forms 31, 41, 51, excluding structurally different branch reports.

`download_supplement.py` downloads reporting context separately, preserving the original financial CSVs. It verifies quarterly totals, dates, and unique keys. A newer metadata fetch can reflect revisions; the committed supplement is the reproducibility snapshot. ACTEVT is a structural-event clue, not a complete history or a definitive disappearance cause.

The primary model retains 2020–2024. This leaves few distinct regimes, even with many bank-quarter rows. All results use revised retrospective reports without release timestamps. The 2024 holdout was already inspected by the earlier project.
