"""Evidence Workbench execution engine (reference implementation, simulated retrieval).

The engine enforces the policy gates (query ceiling, and an authorization grant
for REMOTE_DISPATCH) and records each query in a hash-chained ledger.

The retrieval itself is simulated. No model or dataset is loaded. Query texts
come from a fixed list of ten SciFact-style placeholder claims (cycled when more
than ten queries are requested, so query IDs repeat); they are not read from the
BEIR SciFact dataset. A simulated ranking can contain the same document ID
twice. Retrieved document IDs come from a seeded random generator, and one
"relevant" document is planted at a random rank, or left out of the top k
(2 chances in top_k + 2, so about 1 query in 6 at top_k=10). NDCG@10 and MRR@10
are computed over the first 10 retrieved documents. The resulting NDCG@10 and
MRR@10 values exercise the evidence pipeline. They say nothing about any
model's retrieval quality.
"""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from core.ledger import EvidenceLedger, compute_sha256
from .schema import (
    ExperimentSpec,
    ExecutionPolicy,
    AuthorizationGrant,
    canonical_json_bytes,
)

RETRIEVAL_BACKEND = "simulated-seeded-rng"
SIMULATION_NOTICE = (
    "Simulated retrieval: no model or dataset is loaded. Document IDs come from a "
    "seeded RNG and a relevant document is planted at a random rank (or left out), "
    "so the metrics test the evidence pipeline, not a model."
)


class GovernanceAuthorizationViolation(Exception):
    """Raised when an operation lacks a valid authorization grant (SYS-021); see AuthorizationGrant for its limits."""
    pass


class QuotaExceededViolation(Exception):
    """Raised when an operation attempts to exceed policy-defined query/resource ceilings."""
    pass


def enforce_remote_dispatch_grant(spec: ExperimentSpec, allowed_signers: str | None) -> None:
    """Invariant Gate 2 (SYS-021): remote dispatch needs a grant signed by an allowed signer.

    The grant must name this spec's digest, be signed by a key the allowed-signers
    file lists for the grant's operator, and cover the spec's query count. With
    no allowed-signers file the grant cannot be verified, so the gate refuses.
    """
    policy = spec.execution_policy
    if not (policy.mode == "REMOTE_DISPATCH" or policy.require_authorization_grant):
        return
    spec_digest = spec.canonical_digest()
    if policy.grant is None:
        raise GovernanceAuthorizationViolation(
            f"Remote dispatch rejected: missing authorization grant for payload {spec_digest}."
        )
    if allowed_signers is None:
        raise GovernanceAuthorizationViolation(
            "Remote dispatch rejected: no allowed-signers file is configured, so the grant cannot be verified."
        )
    try:
        valid = policy.grant.verify(spec_digest, allowed_signers)
    except (OSError, ValueError) as e:
        raise GovernanceAuthorizationViolation(f"Remote dispatch rejected: the grant cannot be checked: {e}") from e
    if not valid:
        raise GovernanceAuthorizationViolation(
            f"Remote dispatch rejected: grant does not match payload {spec_digest} or is not signed by an "
            f"allowed signer for operator {policy.grant.operator!r}."
        )
    if spec.query_count > policy.grant.max_query_budget:
        raise GovernanceAuthorizationViolation(
            f"Remote dispatch rejected: {spec.query_count} queries exceed the grant's budget of "
            f"{policy.grant.max_query_budget}."
        )


@dataclass
class QueryRecord:
    query_id: str
    query_text: str
    retrieved_doc_ids: List[str]
    relevance_scores: List[float]  # 0/1 relevance label per retrieved doc, not retriever scores
    ndcg_at_10: float
    mrr_at_10: float
    query_digest: str


@dataclass
class ExecutionResult:
    spec: ExperimentSpec
    spec_digest: str
    evaluated_queries: List[QueryRecord]
    metrics: Dict[str, float]
    chain_digest: str
    ledger_blocks_count: int
    environment_telemetry: Dict[str, Any]
    elapsed_seconds: float
    retrieval_backend: str = RETRIEVAL_BACKEND

    @property
    def merkle_root(self) -> str:
        """Deprecated alias for chain_digest (the value is not a Merkle root)."""
        return self.chain_digest


class EvidenceWorkbenchEngine:
    """Runs a (simulated) benchmark spec behind the policy gates and records it in a ledger."""

    def __init__(self, ledger: EvidenceLedger | None = None, allowed_signers: str | None = None):
        self.ledger = ledger or EvidenceLedger(":memory:")
        self.allowed_signers = allowed_signers

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        """Execute the benchmark specification under governed invariants."""
        start_time = time.perf_counter()

        # Invariant Gate 1: Check Query Ceiling
        policy = spec.execution_policy
        if spec.query_count > policy.max_query_ceiling:
            raise QuotaExceededViolation(
                f"Requested query count ({spec.query_count}) exceeds policy ceiling ({policy.max_query_ceiling})."
            )

        # Invariant Gate 2: Enforce Authorization for Remote Dispatch
        spec_digest = spec.canonical_digest()
        enforce_remote_dispatch_grant(spec, self.allowed_signers)

        # Environment Telemetry Capture
        env_telemetry = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "pid": os.getpid(),
        }

        # Simulated corpus (multi-corpus benchmark simulation): placeholder claims, seeded pseudo-retrieval (see module docstring)
        rng = random.Random(spec.random_seed)
        query_records: List[QueryRecord] = []
        ndcg_list: List[float] = []
        mrr_list: List[float] = []
        ds_lower = spec.dataset.lower()
        if "nfcorpus" in ds_lower:
            corpus_prefix = "nfcorpus"
            corpus_claims = [
                ("q-nfcorpus-001", "Dietary magnesium intake is inversely associated with systemic cardiovascular risk."),
                ("q-nfcorpus-002", "Curcumin suppresses NF-kappaB activation and metastatic signaling in colon carcinoma."),
                ("q-nfcorpus-003", "Anthocyanin-rich berry consumption improves insulin sensitivity in metabolic syndrome."),
                ("q-nfcorpus-004", "Omega-3 polyunsaturated fatty acids attenuate inflammatory cytokine cascades."),
                ("q-nfcorpus-005", "Sulforaphane induces phase II detoxification enzymes via Nrf2 nuclear translocation."),
                ("q-nfcorpus-006", "High dietary fiber intake reduces systemic serum C-reactive protein concentrations."),
                ("q-nfcorpus-007", "Zinc supplementation decreases incidence and duration of pediatric acute diarrhea."),
                ("q-nfcorpus-008", "Polyphenol bioavailability depends on microbial conversion in the human distal gut."),
                ("q-nfcorpus-009", "Vitamin D receptor polymorphisms modulate bone mineral density response to ergocalciferol."),
                ("q-nfcorpus-010", "Resveratrol activates SIRT1 deacetylase pathway to improve endothelial function."),
            ]
        elif "cqadupstack" in ds_lower or "cqa" in ds_lower:
            corpus_prefix = "cqa"
            corpus_claims = [
                ("q-cqa-001", "How to resolve dangling pointer errors in memory-constrained system runtimes?"),
                ("q-cqa-002", "Optimal database indexing strategies for high-frequency append-only ledgers."),
                ("q-cqa-003", "Mitigating quadratic attention bottlenecks in long-context transformer architectures."),
                ("q-cqa-004", "Best practices for implementing fail-closed cryptographic token verification in microservices."),
                ("q-cqa-005", "Configuring loopback socket IPC for low-latency hardware simulation on edge nodes."),
                ("q-cqa-006", "Preventing AST prompt-injection escape vulnerabilities in LLM session memory pipelines."),
                ("q-cqa-007", "How to compute deterministic Merkle roots over canonical RFC 8785 JSON blocks?"),
                ("q-cqa-008", "Synchronizing distributed state machines across unstable hardware dock connections."),
                ("q-cqa-009", "Debugging thermal throttling and frame-drop anomalies in real-time camera capture daemons."),
                ("q-cqa-010", "Strategies for automated rollback recovery when write-ahead logs encounter torn-tail bytes."),
            ]
        else:
            corpus_prefix = "scifact"
            corpus_claims = [
                ("q-scifact-001", "0-dimensional nanomaterials exhibit quantum dot confinement effect."),
                ("q-scifact-002", "Activating transcription factor 3 suppresses lipopolysaccharide induced cytokine production."),
                ("q-scifact-003", "ALDH1A1 expression correlates with poor clinical outcomes in clear cell renal carcinoma."),
                ("q-scifact-004", "Alternative splicing of vascular endothelial growth factor modulates angiogenic potential."),
                ("q-scifact-005", "Amylin receptor agonists reduce voluntary alcohol consumption in rodent models."),
                ("q-scifact-006", "Apelin signaling stimulates myocardial capillary density and angiogenesis."),
                ("q-scifact-007", "Autophagy activation preserves skeletal muscle mass during cancer cachexia."),
                ("q-scifact-008", "Beta-catenin stabilization accelerates hematopoietic stem cell regeneration."),
                ("q-scifact-009", "CD8+ T-cell exhaustion is reversible via dual checkpoint blockade."),
                ("q-scifact-010", "Crispr-Cas9 off-target cleavage is minimized by high-fidelity engineered nucleases."),
            ]

        # Use spec.query_count queries (cycled or sampled deterministically)
        for i in range(spec.query_count):
            qid, claim_text = corpus_claims[i % len(corpus_claims)]
            # Seeded pseudo-retrieval results
            retrieved_docs = [f"doc-{corpus_prefix}-{rng.randint(100, 999)}" for _ in range(spec.top_k)]
            # Relevant document planted deterministically
            relevant_doc = f"doc-{corpus_prefix}-{100 + (i % 7)}"
            rel_rank = rng.randint(1, spec.top_k + 2)
            if rel_rank <= spec.top_k:
                retrieved_docs[rel_rank - 1] = relevant_doc

            # Compute NDCG@10 and MRR@10 over the first 10 documents. The relevant
            # document counts once, at its first rank: a random filler ID can
            # collide with it, and counting both copies pushed NDCG above 1.
            hit_rank = next(
                (idx for idx, d in enumerate(retrieved_docs[:min(spec.top_k, 10)], start=1) if d == relevant_doc),
                None,
            )
            dcg = 1.0 / math.log2(hit_rank + 1) if hit_rank else 0.0
            ndcg = round(dcg, 4)  # IDCG = 1: one relevant document, ideally at rank 1
            mrr = round(1.0 / hit_rank, 4) if hit_rank else 0.0

            ndcg_list.append(ndcg)
            mrr_list.append(mrr)

            q_digest = compute_sha256(f"{qid}:{claim_text}:{','.join(retrieved_docs)}:{ndcg}:{mrr}")
            query_records.append(
                QueryRecord(
                    query_id=qid,
                    query_text=claim_text,
                    retrieved_doc_ids=retrieved_docs,
                    relevance_scores=[1.0 if idx == hit_rank else 0.0 for idx in range(1, len(retrieved_docs) + 1)],
                    ndcg_at_10=ndcg,
                    mrr_at_10=mrr,
                    query_digest=q_digest,
                )
            )

            # Append query block to ledger with deterministic sequence timestamp
            self.ledger.append(
                domain="EVIDENCE_WORKBENCH",
                action="RETRIEVAL_QUERY_EVAL",
                payload_data={"query_id": qid, "query_digest": q_digest, "model": spec.model_id},
                metric_data={"ndcg@10": ndcg, "mrr@10": mrr},
                timestamp=1000.0 + float(i),
            )

        mean_ndcg = round(sum(ndcg_list) / len(ndcg_list), 4) if ndcg_list else 0.0
        mean_mrr = round(sum(mrr_list) / len(mrr_list), 4) if mrr_list else 0.0
        # Corpus Task Complexity: a derived convenience score, not a standard IR metric
        ctc = round(1.0 - (mean_ndcg * 0.8 + mean_mrr * 0.2), 4)

        metrics = {
            "NDCG@10": mean_ndcg,
            "MRR@10": mean_mrr,
            "CTC": ctc,
            "queries_evaluated": len(query_records),
        }

        # Commit final batch consolidation block
        self.ledger.append(
            domain="EVIDENCE_WORKBENCH",
            action="BENCHMARK_BATCH_CONSOLIDATION",
            payload_data={
                "experiment_id": spec.experiment_id,
                "spec_digest": spec_digest,
                "dataset": spec.dataset,
            },
            metric_data=metrics,
            timestamp=2000.0,
        )

        elapsed = time.perf_counter() - start_time
        chain_blocks = self.ledger.fetch_chain()
        chain_digest = self.ledger.chain_digest()

        return ExecutionResult(
            spec=spec,
            spec_digest=spec_digest,
            evaluated_queries=query_records,
            metrics=metrics,
            chain_digest=chain_digest,
            ledger_blocks_count=len(chain_blocks),
            environment_telemetry=env_telemetry,
            elapsed_seconds=round(elapsed, 4),
        )
