"""Canonical schema definitions for DNA-Lang Evidence Workbench."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core import signing

GRANT_NAMESPACE = "dnalang-authorization-grant"


def canonical_json_bytes(data: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, compact separators, UTF-8, NaN rejected.

    Modeled on RFC 8785 (JCS) but not a full implementation: floats use
    Python's repr, and keys sort by code point rather than by UTF-16 units.
    """
    return json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compute_sha256(data: bytes | str) -> str:
    """Compute SHA-256 hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class AuthorizationGrant:
    """Authorization grant bound to a spec digest and a query budget (SYS-021 invariant).

    ``signature`` is an SSH signature (SSHSIG, core/signing.py) made by the key of
    ``operator`` over the other fields. The engine checks it against an
    allowed-signers file, so only a key the verifier lists for that operator can
    authorize a remote dispatch. ``operator`` must be a plain principal name (no
    spaces, commas, quotes or wildcards). ``max_query_budget`` is enforced.
    ``expires_at_iso`` is recorded but not enforced, because a capsule must replay
    the same way later.
    """
    grant_id: str
    operator: str
    target_payload_sha256: str
    max_query_budget: int
    expires_at_iso: str
    signature: str

    def signed_bytes(self) -> bytes:
        """The bytes the signature covers: every field except the signature itself."""
        return canonical_json_bytes({
            "expires_at_iso": self.expires_at_iso,
            "grant_id": self.grant_id,
            "max_query_budget": self.max_query_budget,
            "operator": self.operator,
            "target_payload_sha256": self.target_payload_sha256,
        })

    def verify(self, payload_sha256: str, allowed_signers: str) -> bool:
        """True when the grant is bound to this payload and signed by an allowed signer for its operator.

        Raises ValueError or OSError when the allowed-signers file cannot be read or parsed.
        """
        if self.target_payload_sha256 != payload_sha256:
            return False
        return signing.verify_signature(
            self.signed_bytes(), self.signature, allowed_signers, self.operator, GRANT_NAMESPACE,
        ).ok

    @classmethod
    def issue(cls, key_path: str, *, grant_id: str, operator: str, target_payload_sha256: str,
              max_query_budget: int, expires_at_iso: str) -> "AuthorizationGrant":
        """Create a grant signed with the OpenSSH key at ``key_path`` (see core/signing.py)."""
        signing.check_principal(operator)
        grant = cls(grant_id=grant_id, operator=operator, target_payload_sha256=target_payload_sha256,
                    max_query_budget=max_query_budget, expires_at_iso=expires_at_iso, signature="")
        grant.signature = signing.sign_bytes(grant.signed_bytes(), key_path, GRANT_NAMESPACE)
        return grant


@dataclass
class ExecutionPolicy:
    """Execution policy. The engine checks max_query_ceiling and, for REMOTE_DISPATCH,
    the grant (signature, payload and budget); timeout_seconds is recorded but not enforced."""
    mode: str = "LOCAL_BOUNDED"  # "LOCAL_BOUNDED" or "REMOTE_DISPATCH"
    max_query_ceiling: int = 100
    timeout_seconds: float = 300.0
    require_authorization_grant: bool = False
    grant: Optional[AuthorizationGrant] = None


@dataclass
class ExperimentSpec:
    """Canonical specification for a dense-retrieval benchmark run.

    ``model_id`` and ``dataset`` are labels recorded in the spec. The reference
    engine loads neither (see workbench/engine.py). Real runs
    (workbench/real_scifact.py) use ``model_id`` as the embedding model and
    read the backend, model revision, query prefix and reranker from
    ``retrieval_options``; it joins the canonical digest only when non-empty,
    so specs without it keep their existing digests.
    """
    experiment_id: str
    model_id: str
    benchmark_suite: str
    dataset: str
    split: str
    query_count: int
    top_k: int = 10
    similarity_metric: str = "cosine"
    random_seed: int = 42
    target_metrics: List[str] = field(default_factory=lambda: ["NDCG@10", "MRR@10", "CTC"])
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    retrieval_options: Dict[str, Any] = field(default_factory=dict)
    created_at_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to structured dict."""
        return asdict(self)

    def canonical_digest(self) -> str:
        """SHA-256 of the canonical JSON of this spec's identity fields."""
        d = self.to_dict()
        # Exclude variable timestamp from content identity
        identity_subset = {
            "experiment_id": d["experiment_id"],
            "model_id": d["model_id"],
            "benchmark_suite": d["benchmark_suite"],
            "dataset": d["dataset"],
            "split": d["split"],
            "query_count": d["query_count"],
            "top_k": d["top_k"],
            "similarity_metric": d["similarity_metric"],
            "random_seed": d["random_seed"],
            "target_metrics": sorted(d["target_metrics"]),
            "policy_mode": d["execution_policy"]["mode"],
            "max_query_ceiling": d["execution_policy"]["max_query_ceiling"],
        }
        if d["retrieval_options"]:
            identity_subset["retrieval_options"] = d["retrieval_options"]
        return compute_sha256(canonical_json_bytes(identity_subset))
