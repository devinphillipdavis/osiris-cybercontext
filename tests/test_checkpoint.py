"""Tests for ledger checkpoints: truncation and rewrite detection, and signed checkpoint documents."""

import os
import shutil
import subprocess
import tempfile
import unittest

from core.ledger import EvidenceLedger, compute_sha256
from workbench.checkpoint import create_checkpoint, verify_checkpoint

HAVE_SSH_KEYGEN = shutil.which("ssh-keygen") is not None


def _ledger(blocks=5):
    ledger = EvidenceLedger(":memory:")
    for i in range(blocks):
        ledger.append("domain", f"action-{i}", {"i": i}, {"metric": i}, timestamp=1000.0 + i)
    return ledger


def _rewrite_and_rechain(ledger, entry_id, metric_json):
    """Act as the database's writer: change one block and recompute every hash after it."""
    ledger.conn.execute("UPDATE ledger SET metric_json = ? WHERE entry_id = ?", (metric_json, entry_id))
    previous = ledger.conn.execute(
        "SELECT current_hash FROM ledger WHERE entry_id < ? ORDER BY entry_id DESC LIMIT 1", (entry_id,)
    ).fetchone()
    prev_hash = previous[0] if previous else EvidenceLedger.GENESIS_HASH
    rows = ledger.conn.execute(
        "SELECT entry_id, timestamp, domain, action, payload_digest, metric_json FROM ledger "
        "WHERE entry_id >= ? ORDER BY entry_id",
        (entry_id,),
    ).fetchall()
    for row_id, timestamp, domain, action, payload_digest, metric in rows:
        current = compute_sha256(f"{prev_hash}:{row_id}:{timestamp:.4f}:{domain}:{action}:{payload_digest}:{metric}")
        ledger.conn.execute(
            "UPDATE ledger SET prev_hash = ?, current_hash = ? WHERE entry_id = ?", (prev_hash, current, row_id)
        )
        prev_hash = current
    ledger.conn.commit()


class TestLedgerCheckpoint(unittest.TestCase):
    def test_intact_ledger_matches_its_checkpoint(self):
        ledger = _ledger()
        ok, message = ledger.check_checkpoint(ledger.checkpoint())
        self.assertTrue(ok, message)

    def test_blocks_added_after_the_checkpoint_still_extend_it(self):
        ledger = _ledger()
        checkpoint = ledger.checkpoint()
        ledger.append("domain", "later", {"x": 1}, {"metric": 9}, timestamp=2000.0)
        ok, message = ledger.check_checkpoint(checkpoint)
        self.assertTrue(ok, message)

    def test_deleted_newest_block_is_detected(self):
        ledger = _ledger()
        checkpoint = ledger.checkpoint()
        ledger.conn.execute("DELETE FROM ledger WHERE entry_id = 5")
        ledger.conn.commit()
        self.assertTrue(ledger.verify_integrity()[0])  # the plain chain check cannot see this
        ok, message = ledger.check_checkpoint(checkpoint)
        self.assertFalse(ok)
        self.assertIn("newest blocks were deleted", message)

    def test_history_rewritten_with_a_recomputed_chain_is_detected(self):
        ledger = _ledger()
        checkpoint = ledger.checkpoint()
        _rewrite_and_rechain(ledger, 2, '{"metric":99}')
        self.assertTrue(ledger.verify_integrity()[0])  # every hash was recomputed, so the chain looks intact
        ok, message = ledger.check_checkpoint(checkpoint)
        self.assertFalse(ok)
        self.assertIn("history was rewritten", message)

    def test_direct_edit_without_rehashing_fails_the_integrity_check(self):
        ledger = _ledger()
        checkpoint = ledger.checkpoint()
        ledger.conn.execute("UPDATE ledger SET metric_json = '{\"metric\":99}' WHERE entry_id = 3")
        ledger.conn.commit()
        ok, message = ledger.check_checkpoint(checkpoint)
        self.assertFalse(ok)
        self.assertIn("Data tampering", message)

    def test_empty_ledger_checkpoint_is_extended_by_later_blocks(self):
        ledger = EvidenceLedger(":memory:")
        checkpoint = ledger.checkpoint()
        self.assertEqual(checkpoint["block_count"], 0)
        self.assertTrue(ledger.check_checkpoint(checkpoint)[0])
        ledger.append("domain", "first", {}, {}, timestamp=1.0)
        self.assertTrue(ledger.check_checkpoint(checkpoint)[0])

    def test_malformed_checkpoint_raises(self):
        ledger = _ledger()
        with self.assertRaises(ValueError):
            ledger.check_checkpoint({"block_count": "5", "head_hash": "x", "chain_digest": "y"})

    def test_read_only_ledger_needs_a_file_and_refuses_writes(self):
        with self.assertRaises(ValueError):
            EvidenceLedger(":memory:", read_only=True)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "ledger.db")
        writer = EvidenceLedger(path)
        writer.append("domain", "action", {}, {}, timestamp=1.0)
        writer.close()
        reader = EvidenceLedger(path, read_only=True)
        self.addCleanup(reader.close)
        self.assertEqual(reader.checkpoint()["block_count"], 1)
        with self.assertRaises(Exception):
            reader.append("domain", "refused", {}, {}, timestamp=2.0)


@unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
class TestSignedCheckpoints(unittest.TestCase):
    NAMESPACE = "dnalang-ledger-checkpoint"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.key = self._make_key("key")
        self.other_key = self._make_key("other")
        with open(self.key + ".pub", encoding="utf-8") as f:
            pub = f.read().strip()
        self.allowed = os.path.join(self.dir, "allowed_signers")
        with open(self.allowed, "w", encoding="utf-8") as f:
            f.write(f'custodian namespaces="{self.NAMESPACE}" {pub}\n')

    def _make_key(self, name):
        path = os.path.join(self.dir, name)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", path], check=True)
        return path

    def test_signed_checkpoint_passes_on_the_ledger_it_came_from(self):
        ledger = _ledger()
        document = create_checkpoint(ledger, self.key)
        ledger.append("domain", "later", {}, {}, timestamp=3000.0)
        ok, message = verify_checkpoint(ledger, document, self.allowed, "custodian")
        self.assertTrue(ok, message)

    def test_altered_checkpoint_body_fails_the_signature(self):
        ledger = _ledger()
        document = create_checkpoint(ledger, self.key)
        document["checkpoint"]["block_count"] = 4
        ok, message = verify_checkpoint(ledger, document, self.allowed, "custodian")
        self.assertFalse(ok)
        self.assertIn("INVALID", message)

    def test_checkpoint_signed_by_an_unlisted_key_is_untrusted(self):
        ledger = _ledger()
        document = create_checkpoint(ledger, self.other_key)
        ok, message = verify_checkpoint(ledger, document, self.allowed, "custodian")
        self.assertFalse(ok)
        self.assertIn("UNTRUSTED", message)

    def test_unsigned_document_is_rejected(self):
        ledger = _ledger()
        with self.assertRaises(ValueError):
            verify_checkpoint(ledger, {"checkpoint": ledger.checkpoint()}, self.allowed, "custodian")


if __name__ == "__main__":
    unittest.main()
