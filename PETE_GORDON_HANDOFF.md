# Handoff Note for Pete Gordon

**Subject:** Zero-dependency evidence ledger & replay capsules for Olmo 3 / CyberContext

Hi Pete,

I know you are neck-deep in MTEB/BEIR benchmark evaluation with Olmo 3 while balancing CyberContext and Tandemonium.

Over the past few weeks, I built a lightweight, zero-dependency companion prototype called `osiris-cybercontext`:
https://github.com/osiris-dnalang/osiris-cybercontext

### Why this might save you time:
When evaluating models across long contexts or benchmarks, three problems kept biting me:
1. **Benchmark Drift:** Ingestion pipelines change slightly between runs, making NDCG/MRR comparisons non-reproducible across machines.
2. **Context Poisoning:** Prompt injection or malformed turns slip into session context silently.
3. **Audit Overhead:** Proving to external reviewers that an evaluation run actually matched the reported scores without shipping a multi-gigabyte database.

### What it does (Standard Library Python 3.10+ Only):
- **Deterministic Replay Capsules (`.evidence.zip`):** Packages query inputs, ranked outputs, and exact scoring matrices with SHA-256 Merkle verification. Anyone can run `python -m workbench.replay <capsule>` and verify the exact run locally in under 2 seconds.
- **AST/Regex Turn Firewall:** Pre-execution screening for session memory to block unauthorized payload execution before context compaction.
- **BEIR SciFact Adapter:** Plugs directly into standard BEIR retrieval tasks, scoring BM25 or embedding models, with an optional Olmo 3 instruction reranker.

You can clone it and run the verified demo right now:
```bash
git clone https://github.com/osiris-dnalang/osiris-cybercontext
cd osiris-cybercontext
python demo.py
```

I documented the exact boundaries of what is real vs. simulated in `README.md` and `ERRATA.md`—no black boxes, no external pip dependencies required for the core ledger.

If this looks useful for your Olmo 3 evaluation runs or CyberContext verification, I would love to get your thoughts on it. If there is a clean place to integrate it into your active pipeline, I’m available to help wire it up.

Best,  
Devin Phillip Davis  
Agile Defense Systems LLC  
devin.dnalang@outlook.com
