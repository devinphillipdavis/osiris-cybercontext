"""Tamper-evident evidence ledger (SQLite + SHA-256 hash chain).

Zero-dependency append-only log used by the demo and the evidence workbench.

What verify_integrity() detects: most edits to a stored block that don't also
recompute its hash and every later hash (for example a direct SQL UPDATE of a
metric), and deleted or re-ordered blocks before the newest one.

What it does not detect:
- Deleting the newest block(s), or all of them. Only check_checkpoint() shows
  this, and only against a checkpoint taken earlier and kept outside the database.
- Timestamp edits smaller than 1e-4 s, because the hash covers the timestamp
  rounded to 4 decimals, and edits that move text across the ':' separators
  between hashed fields.
- A rewrite of any block followed by recomputing the rest of the chain, which
  anyone with write access to the database can do. check_checkpoint() catches
  it for any head that was checkpointed before the rewrite.
A passing verify_integrity() means "this file is internally consistent", not
"this history is authentic". Authenticity needs a signed checkpoint whose key and
storage are kept away from the database's writer (workbench/checkpoint.py).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def canonical_json(data: Any) -> str:
    """Serialize to deterministic JSON: sorted keys, compact separators.

    Modeled on RFC 8785 (JCS) but not a full implementation: number formatting
    and the ordering of some non-ASCII keys can differ, and values JSON can't
    represent are stringified via ``default=str``.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_sha256(text: str | bytes) -> str:
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


@dataclass
class LedgerRecord:
    entry_id: int
    prev_hash: str
    timestamp: float
    domain: str
    action: str
    payload_digest: str
    metric_data: Dict[str, Any]
    current_hash: str


class EvidenceLedger:
    """SQLite-backed append-only hash chain."""

    GENESIS_HASH = "0" * 64

    def __init__(self, db_path: str = ":memory:", read_only: bool = False):
        self.db_path = db_path
        if read_only:
            # A checker must not create or change the evidence it reads.
            if db_path == ":memory:":
                raise ValueError("a read-only ledger needs a database file")
            self.conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.conn = sqlite3.connect(self.db_path)
            self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prev_hash TEXT NOT NULL,
                timestamp REAL NOT NULL,
                domain TEXT NOT NULL,
                action TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                metric_json TEXT NOT NULL,
                current_hash TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def get_latest_hash(self) -> Tuple[int, str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT entry_id, current_hash FROM ledger ORDER BY entry_id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return row[0], row[1]
        return 0, self.GENESIS_HASH

    def append(
        self,
        domain: str,
        action: str,
        payload_data: Any,
        metric_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> LedgerRecord:
        last_id, prev_hash = self.get_latest_hash()
        new_id = last_id + 1
        now = timestamp if timestamp is not None else time.time()

        payload_digest = compute_sha256(canonical_json(payload_data))
        metrics = metric_data or {}
        metric_str = canonical_json(metrics)

        preimage = f"{prev_hash}:{new_id}:{now:.4f}:{domain}:{action}:{payload_digest}:{metric_str}"
        current_hash = compute_sha256(preimage)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ledger (entry_id, prev_hash, timestamp, domain, action, payload_digest, metric_json, current_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (new_id, prev_hash, now, domain, action, payload_digest, metric_str, current_hash))
        self.conn.commit()

        return LedgerRecord(
            entry_id=new_id,
            prev_hash=prev_hash,
            timestamp=now,
            domain=domain,
            action=action,
            payload_digest=payload_digest,
            metric_data=metrics,
            current_hash=current_hash,
        )

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM ledger ORDER BY entry_id ASC")
        rows = cursor.fetchall()
        self.conn.row_factory = None

        if not rows:
            return True, "Ledger is empty (valid genesis state)."

        expected_prev = self.GENESIS_HASH
        for idx, row in enumerate(rows, start=1):
            if row["entry_id"] != idx:
                return False, f"Sequence break: expected #{idx}, got #{row['entry_id']}"
            if row["prev_hash"] != expected_prev:
                return False, f"Chain linkage break at block #{idx}"
            if not isinstance(row["timestamp"], (int, float)):
                return False, f"Malformed block #{idx}: timestamp is not a number"

            preimage = f"{row['prev_hash']}:{row['entry_id']}:{row['timestamp']:.4f}:{row['domain']}:{row['action']}:{row['payload_digest']}:{row['metric_json']}"
            recomputed = compute_sha256(preimage)
            if recomputed != row["current_hash"]:
                return False, f"Data tampering detected at block #{idx}: computed {recomputed[:12]} != stored {row['current_hash'][:12]}"

            expected_prev = row["current_hash"]

        return True, f"All {len(rows)} blocks verified successfully. Hash chain is intact."

    def chain_digest(self) -> str:
        """SHA-256 over the ordered block hashes joined with ':'.

        One digest that summarizes the whole chain. It is not a Merkle tree
        root: there is no tree and no inclusion proof.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT current_hash FROM ledger ORDER BY entry_id ASC")
        hashes = [r[0] for r in cursor.fetchall()]

        if not hashes:
            return self.GENESIS_HASH
        combined = ":".join(hashes)
        return compute_sha256(combined)

    def checkpoint(self) -> Dict[str, Any]:
        """The chain head: block count, head hash and chain digest, for keeping outside the database."""
        block_count, head_hash = self.get_latest_hash()
        return {"block_count": block_count, "head_hash": head_hash, "chain_digest": self.chain_digest()}

    def check_checkpoint(self, checkpoint: Dict[str, Any]) -> Tuple[bool, str]:
        """Check that this ledger still extends a checkpoint taken earlier.

        Fails when the chain is internally broken (verify_integrity), when the
        ledger has fewer blocks than the checkpoint (newest blocks were deleted),
        when the block at the checkpoint's height has another hash (history was
        rewritten after the checkpoint), or when the digest over the checkpointed
        blocks differs. Blocks added after the checkpoint are allowed.
        """
        count = checkpoint.get("block_count")
        head_hash = checkpoint.get("head_hash")
        digest = checkpoint.get("chain_digest")
        if (not isinstance(count, int) or isinstance(count, bool) or count < 0
                or not isinstance(head_hash, str) or not isinstance(digest, str)):
            raise ValueError("malformed checkpoint: needs block_count, head_hash and chain_digest")

        ok, message = self.verify_integrity()
        if not ok:
            return False, message
        hashes = [row[0] for row in self.conn.execute("SELECT current_hash FROM ledger ORDER BY entry_id ASC")]
        if len(hashes) < count:
            return False, (f"ledger has {len(hashes)} blocks, fewer than the checkpoint's {count}: "
                           "the newest blocks were deleted")
        anchored = hashes[:count]
        if (anchored[-1] if anchored else self.GENESIS_HASH) != head_hash:
            return False, f"block #{count} does not match the checkpoint: history was rewritten after it was taken"
        if (compute_sha256(":".join(anchored)) if anchored else self.GENESIS_HASH) != digest:
            return False, "the chain digest over the checkpointed blocks does not match the checkpoint"
        return True, f"ledger extends the checkpoint: {len(hashes)} blocks, block #{count} matches"

    def get_merkle_root(self) -> str:
        """Deprecated alias for chain_digest(); the value is not a Merkle root."""
        return self.chain_digest()

    def merkle_root(self) -> str:
        """Deprecated alias for chain_digest(); the value is not a Merkle root."""
        return self.chain_digest()

    def fetch_chain(self) -> List[Dict[str, Any]]:
        """Fetch all ledger blocks as dictionaries."""
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM ledger ORDER BY entry_id ASC")
        rows = [dict(r) for r in cursor.fetchall()]
        self.conn.row_factory = None
        return rows

    def close(self):
        self.conn.close()
