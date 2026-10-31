# osiris-cybercontext v0.3.0 — Release Notes (2026-10-31)

A major benchmark integrity and backend retrieval release integrating the **Real SciFact Backend**, **`EXP-OLMO3-BEIR-SCIFACT-001` Benchmark Evidence Capsule**, and multi-corpus evaluation across NFCorpus and CQADupStack.

---

## 1. Real SciFact Backend & Multi-Corpus Evaluation (`workbench/real_scifact.py`)
- Replaces mock evaluation stubs with live, end-to-end evaluation against real SciFact abstract retrieval and claims verification datasets.
- Implements deterministic evidence ranking, MRR, nDCG@10, and precision@k metrics.
- Extends the evaluation pipeline across BEIR corpora: **SciFact**, **NFCorpus**, and **CQADupStack**.

## 2. Benchmark Evidence Capsule (`benchmark_capsule.evidence.zip`)
- Integrates the formal evidence capsule for `EXP-OLMO3-BEIR-SCIFACT-001`.
- Cryptographically hashes all ground-truth queries, retrieval runs, and evaluation metrics into the append-only evidence ledger.
- Enforces strict secret scrubbing, ensuring zero API tokens or local environment credentials leak into published capsules.

## 3. Governance & Invariants (SYS-021)
- Integrates the `CyberContextASTFirewall` with payload-bound authorization receipts ($\mathcal{H}_{\text{payload}} = \text{SHA-256}(\text{code})$).
- Validates all candidate evaluation turn scripts through fail-closed AST gates prior to local execution.

---

## 4. Lineage & Identifiers
- **Software Repository**: [https://github.com/osiris-dnalang/osiris-cybercontext](https://github.com/osiris-dnalang/osiris-cybercontext)
- **Predecessor Releases**: `v0.1.0` (2026-10-11), `v0.2.0` (2026-10-15)
- **Zenodo Concept**: [`10.5281/zenodo.23305203`](https://doi.org/10.5281/zenodo.23305203)
