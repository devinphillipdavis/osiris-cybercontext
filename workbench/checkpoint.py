"""Signed ledger checkpoints: a copy of a ledger's head that its database writer cannot rewrite.

A checkpoint records the block count, head hash and chain digest of an
EvidenceLedger. It is signed with an OpenSSH Ed25519 key (core/signing.py) and
kept outside the database. Checking it later shows whether the ledger was
truncated or rewritten after the checkpoint was taken. The protection holds only
while the signing key and the stored checkpoint stay away from whoever writes the
database.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from core import signing
from core.ledger import EvidenceLedger, canonical_json

CHECKPOINT_NAMESPACE = "dnalang-ledger-checkpoint"


def create_checkpoint(ledger: EvidenceLedger, key_path: str) -> Dict[str, Any]:
    """The ledger's head, signed with ``key_path``, as a JSON-ready document."""
    body = ledger.checkpoint()
    signature = signing.sign_bytes(canonical_json(body).encode("utf-8"), key_path, CHECKPOINT_NAMESPACE)
    return {"checkpoint": body, "signature": signature}


def verify_checkpoint(ledger: EvidenceLedger, document: Dict[str, Any], allowed_signers: str,
                      identity: str) -> Tuple[bool, str]:
    """Check a signed checkpoint, then check that ``ledger`` still extends it. Returns (passed, explanation)."""
    body = document.get("checkpoint")
    signature = document.get("signature")
    if not isinstance(body, dict) or not isinstance(signature, str):
        raise ValueError("not a signed checkpoint: needs 'checkpoint' and 'signature'")
    verdict = signing.verify_signature(
        canonical_json(body).encode("utf-8"), signature, allowed_signers, identity, CHECKPOINT_NAMESPACE,
    )
    if not verdict.ok:
        return False, f"checkpoint signature {verdict.status}: {verdict.detail}"
    return ledger.check_checkpoint(body)
