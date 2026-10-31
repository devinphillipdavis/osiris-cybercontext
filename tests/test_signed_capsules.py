"""Tests for signed evidence capsules: sign, verify, forgery and stripping, and the CLI exit codes."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from core import signing
from workbench.capsule import EvidenceCapsule, SIGNATURE_NAME
from workbench.engine import EvidenceWorkbenchEngine
from workbench.schema import ExperimentSpec, canonical_json_bytes, compute_sha256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_SSH_KEYGEN = shutil.which("ssh-keygen") is not None
NAMESPACE = "dnalang-evidence-capsule"


def _build_capsule(path):
    spec = ExperimentSpec(
        experiment_id="EXP-SIGNED-001", model_id="label", benchmark_suite="suite",
        dataset="SciFact", split="test", query_count=2,
    )
    engine = EvidenceWorkbenchEngine()
    try:
        result = engine.execute(spec)
    finally:
        engine.ledger.close()
    return EvidenceCapsule.pack(result, path)


def _read_entries(path):
    with zipfile.ZipFile(path, "r") as zf:
        return [(info.filename, zf.read(info.filename)) for info in zf.infolist()]


def _write_entries(path, entries):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)


@unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
class TestSignedCapsules(unittest.TestCase):
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
            f.write(f'alice namespaces="{NAMESPACE}" {pub}\n')
        self.capsule = _build_capsule(os.path.join(self.dir, "capsule.evidence.zip"))

    def _make_key(self, name):
        path = os.path.join(self.dir, name)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", path], check=True)
        return path

    def test_signed_capsule_verifies_and_still_unpacks(self):
        fingerprint = EvidenceCapsule.sign(self.capsule, self.key)
        verdict = EvidenceCapsule.verify_signature(self.capsule, self.allowed, "alice")
        self.assertEqual(verdict.status, signing.SIGNATURE_VALID, verdict.detail)
        self.assertEqual(verdict.fingerprint, fingerprint)
        self.assertIn(SIGNATURE_NAME, dict(_read_entries(self.capsule)))
        self.assertEqual(EvidenceCapsule.unpack(self.capsule)["manifest"]["experiment_id"], "EXP-SIGNED-001")

    def test_unsigned_capsule_reports_unsigned(self):
        verdict = EvidenceCapsule.verify_signature(self.capsule, self.allowed, "alice")
        self.assertEqual(verdict.status, signing.SIGNATURE_UNSIGNED)

    def test_capsule_cannot_be_signed_twice(self):
        EvidenceCapsule.sign(self.capsule, self.key)
        with self.assertRaises(ValueError):
            EvidenceCapsule.sign(self.capsule, self.other_key)

    def test_signature_from_an_unlisted_key_is_untrusted(self):
        EvidenceCapsule.sign(self.capsule, self.other_key)
        verdict = EvidenceCapsule.verify_signature(self.capsule, self.allowed, "alice")
        self.assertEqual(verdict.status, signing.SIGNATURE_UNTRUSTED)

    def test_rewritten_summary_and_manifest_break_the_signature(self):
        EvidenceCapsule.sign(self.capsule, self.key)
        entries = dict(_read_entries(self.capsule))
        summary = entries["summary.md"].decode("utf-8").replace("NDCG@10:", "NDCG@10 (edited):")
        manifest = json.loads(entries["manifest.json"])
        manifest["manifest_files"]["summary.md"] = {
            "sha256": compute_sha256(summary.encode("utf-8")), "bytes": len(summary.encode("utf-8")),
        }
        rewritten = [
            ("manifest.json", canonical_json_bytes(manifest)),
            ("raw_queries.json", entries["raw_queries.json"]),
            ("summary.md", summary.encode("utf-8")),
            (SIGNATURE_NAME, entries[SIGNATURE_NAME]),
        ]
        _write_entries(self.capsule, rewritten)
        EvidenceCapsule.unpack(self.capsule)  # the checksums now agree with each other
        verdict = EvidenceCapsule.verify_signature(self.capsule, self.allowed, "alice")
        self.assertEqual(verdict.status, signing.SIGNATURE_INVALID, verdict.detail)

    def test_edited_file_without_manifest_update_fails_before_the_signature_check(self):
        EvidenceCapsule.sign(self.capsule, self.key)
        entries = _read_entries(self.capsule)
        entries = [(n, d.replace(b"NDCG@10", b"NDCG@11") if n == "summary.md" else d) for n, d in entries]
        _write_entries(self.capsule, entries)
        with self.assertRaises(ValueError):
            EvidenceCapsule.verify_signature(self.capsule, self.allowed, "alice")

    def test_stripped_signature_reads_as_unsigned(self):
        EvidenceCapsule.sign(self.capsule, self.key)
        entries = [(n, d) for n, d in _read_entries(self.capsule) if n != SIGNATURE_NAME]
        _write_entries(self.capsule, entries)
        verdict = EvidenceCapsule.verify_signature(self.capsule, self.allowed, "alice")
        self.assertEqual(verdict.status, signing.SIGNATURE_UNSIGNED)

    def test_extra_file_is_still_rejected(self):
        EvidenceCapsule.sign(self.capsule, self.key)
        entries = _read_entries(self.capsule) + [("notes.txt", b"extra")]
        _write_entries(self.capsule, entries)
        with self.assertRaises(ValueError):
            EvidenceCapsule.unpack(self.capsule)

    def test_failed_signing_leaves_the_capsule_unchanged(self):
        before = _read_entries(self.capsule)
        with self.assertRaises(RuntimeError):
            EvidenceCapsule.sign(self.capsule, os.path.join(self.dir, "no-such-key"))
        self.assertEqual(_read_entries(self.capsule), before)
        self.assertFalse(os.path.exists(self.capsule + ".signing"))

    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "workbench.cli", *args], cwd=ROOT, capture_output=True, text=True,
        )

    def test_cli_sign_then_verify_exit_codes(self):
        signed = self._cli("sign", self.capsule, "--key", self.key)
        self.assertEqual(signed.returncode, 0, signed.stdout + signed.stderr)
        ok = self._cli("verify", self.capsule, "--allowed-signers", self.allowed, "--identity", "alice",
                       "--require-signature")
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn("VALID", ok.stdout)
        untrusted = self._cli("verify", self.capsule, "--allowed-signers", self.allowed, "--identity", "mallory")
        self.assertEqual(untrusted.returncode, 1)
        self.assertIn("UNTRUSTED", untrusted.stdout)

    def test_cli_require_signature_fails_an_unsigned_capsule(self):
        result = self._cli("verify", self.capsule, "--allowed-signers", self.allowed, "--identity", "alice",
                           "--require-signature")
        self.assertEqual(result.returncode, 1)
        self.assertIn("UNSIGNED", result.stdout)

    def test_cli_rejects_identity_missing_with_signers(self):
        result = self._cli("verify", self.capsule, "--allowed-signers", self.allowed)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--identity is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
