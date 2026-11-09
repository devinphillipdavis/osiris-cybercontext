"""Tests for signed authorization grants (SYS-021): issuing, verifying, the budget, and the CLI round trip."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from core.ledger import compute_sha256
from workbench.engine import EvidenceWorkbenchEngine, GovernanceAuthorizationViolation
from workbench.schema import GRANT_NAMESPACE, AuthorizationGrant, ExecutionPolicy, ExperimentSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_SSH_KEYGEN = shutil.which("ssh-keygen") is not None


def _remote_spec(query_count=5):
    return ExperimentSpec(
        experiment_id="EXP-GRANT", model_id="label", benchmark_suite="MTEB/BEIR", dataset="SciFact",
        split="test", query_count=query_count,
        execution_policy=ExecutionPolicy(mode="REMOTE_DISPATCH", require_authorization_grant=True),
    )


@unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
class TestGrantSignatures(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.key = self._make_key("operator")
        self.other_key = self._make_key("intruder")
        with open(self.key + ".pub", encoding="utf-8") as f:
            pub = f.read().strip()
        self.allowed = os.path.join(self.dir, "allowed_signers")
        with open(self.allowed, "w", encoding="utf-8") as f:
            f.write(f'devin namespaces="{GRANT_NAMESPACE}" {pub}\n')
        self.digest = _remote_spec().canonical_digest()

    def _make_key(self, name):
        path = os.path.join(self.dir, name)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", path], check=True)
        return path

    def _issue(self, key=None, operator="devin", budget=100, digest=None):
        return AuthorizationGrant.issue(
            key or self.key, grant_id="G-1", operator=operator,
            target_payload_sha256=digest or self.digest, max_query_budget=budget,
            expires_at_iso="2099-01-01T00:00:00Z",
        )

    def test_issued_grant_verifies_for_its_operator_and_payload(self):
        self.assertTrue(self._issue().verify(self.digest, self.allowed))

    def test_grant_for_another_payload_fails(self):
        grant = self._issue(digest="0" * 64)
        self.assertFalse(grant.verify(self.digest, self.allowed))

    def test_altered_budget_breaks_the_signature(self):
        grant = self._issue()
        grant.max_query_budget = 1000
        self.assertFalse(grant.verify(self.digest, self.allowed))

    def test_grant_signed_by_an_unlisted_key_is_refused(self):
        self.assertFalse(self._issue(key=self.other_key).verify(self.digest, self.allowed))

    def test_grant_naming_another_operator_is_refused(self):
        self.assertFalse(self._issue(operator="mallory").verify(self.digest, self.allowed))

    def test_legacy_unkeyed_grant_is_refused(self):
        grant = AuthorizationGrant(
            grant_id="G-OLD", operator="devin", target_payload_sha256=self.digest, max_query_budget=100,
            expires_at_iso="2099-01-01T00:00:00Z",
            signature=compute_sha256(f"devin:{self.digest}:100"),
        )
        self.assertFalse(grant.verify(self.digest, self.allowed))

    def test_operator_must_be_a_plain_principal_name(self):
        with self.assertRaises(ValueError):
            self._issue(operator="Devin Phillip Davis")

    def test_engine_enforces_the_query_budget(self):
        spec = _remote_spec(query_count=5)
        spec.execution_policy.grant = AuthorizationGrant.issue(
            self.key, grant_id="G-2", operator="devin", target_payload_sha256=spec.canonical_digest(),
            max_query_budget=3, expires_at_iso="2099-01-01T00:00:00Z",
        )
        engine = EvidenceWorkbenchEngine(allowed_signers=self.allowed)
        self.addCleanup(engine.ledger.close)
        with self.assertRaisesRegex(GovernanceAuthorizationViolation, "exceed the grant's budget"):
            engine.execute(spec)

    def test_engine_without_allowed_signers_refuses_remote_dispatch(self):
        spec = _remote_spec()
        spec.execution_policy.grant = self._issue(digest=spec.canonical_digest())
        engine = EvidenceWorkbenchEngine()
        self.addCleanup(engine.ledger.close)
        with self.assertRaisesRegex(GovernanceAuthorizationViolation, "no allowed-signers file"):
            engine.execute(spec)

    def test_unreadable_allowed_signers_refuses_remote_dispatch(self):
        spec = _remote_spec()
        spec.execution_policy.grant = self._issue(digest=spec.canonical_digest())
        engine = EvidenceWorkbenchEngine(allowed_signers=os.path.join(self.dir, "absent"))
        self.addCleanup(engine.ledger.close)
        with self.assertRaisesRegex(GovernanceAuthorizationViolation, "cannot be checked"):
            engine.execute(spec)


@unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
class TestGrantCommandLine(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.key = os.path.join(self.dir, "operator")
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "operator", "-f", self.key], check=True)
        with open(self.key + ".pub", encoding="utf-8") as f:
            pub = f.read().strip()
        self.allowed = os.path.join(self.dir, "allowed_signers")
        with open(self.allowed, "w", encoding="utf-8") as f:
            f.write(f'devin namespaces="{GRANT_NAMESPACE}" {pub}\n')
        self.grant_path = os.path.join(self.dir, "grant.json")
        self.capsule = os.path.join(self.dir, "remote.evidence.zip")

    def _cli(self, *args):
        return subprocess.run([sys.executable, "-m", "workbench.cli", *args], cwd=ROOT, capture_output=True, text=True)

    def test_issue_dispatch_and_replay_round_trip(self):
        run_args = ["run", "--mode", "REMOTE_DISPATCH", "--queries", "3"]
        refused = self._cli(*run_args, "--out", self.capsule)
        self.assertEqual(refused.returncode, 1)
        self.assertIn("FAIL-CLOSED", refused.stdout)
        digest = re.search(r"Canonical Spec Digest: ([0-9a-f]{64})", refused.stdout).group(1)

        issued = self._cli("grant", "--key", self.key, "--operator", "devin", "--grant-id", "G-CLI",
                           "--payload-sha256", digest, "--budget", "10",
                           "--expires", "2099-01-01T00:00:00Z", "--out", self.grant_path)
        self.assertEqual(issued.returncode, 0, issued.stdout + issued.stderr)
        with open(self.grant_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["target_payload_sha256"], digest)

        dispatched = self._cli(*run_args, "--grant", self.grant_path, "--allowed-signers", self.allowed,
                               "--out", self.capsule)
        self.assertEqual(dispatched.returncode, 0, dispatched.stdout + dispatched.stderr)

        replayed = self._cli("replay", self.capsule, "--allowed-signers", self.allowed)
        self.assertEqual(replayed.returncode, 0, replayed.stdout + replayed.stderr)
        self.assertIn("EXACT_MATCH", replayed.stdout)

        unchecked = self._cli("replay", self.capsule)
        self.assertEqual(unchecked.returncode, 1)
        self.assertIn("NOT_REPLAYABLE", unchecked.stdout)

    def test_grant_command_rejects_a_malformed_digest(self):
        result = self._cli("grant", "--key", self.key, "--operator", "devin", "--grant-id", "G",
                           "--payload-sha256", "not-a-digest", "--budget", "1",
                           "--expires", "2099-01-01T00:00:00Z", "--out", self.grant_path)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(os.path.exists(self.grant_path))


if __name__ == "__main__":
    unittest.main()
