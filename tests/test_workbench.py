"""Unit tests for DNA-Lang Evidence Workbench."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import warnings
import zipfile

from workbench.schema import (
    GRANT_NAMESPACE,
    ExperimentSpec,
    ExecutionPolicy,
    AuthorizationGrant,
    canonical_json_bytes,
    compute_sha256,
)
from workbench.engine import (
    EvidenceWorkbenchEngine,
    GovernanceAuthorizationViolation,
    QuotaExceededViolation,
)
from workbench.capsule import EvidenceCapsule
from workbench.replay import ReplayEngine, ReplayStatus


HAVE_SSH_KEYGEN = shutil.which("ssh-keygen") is not None


def _execute(spec):
    """Run a spec in a fresh engine and close its in-memory ledger."""
    engine = EvidenceWorkbenchEngine()
    try:
        return engine.execute(spec)
    finally:
        engine.ledger.close()


def _ssh_key(directory, name):
    """A throwaway OpenSSH Ed25519 key; returns the private key path."""
    path = os.path.join(directory, name)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", path], check=True)
    return path


def _allowed_signers(directory, key_path, identity, namespace):
    """An allowed-signers file that trusts ``key_path`` for ``identity`` in ``namespace``."""
    with open(key_path + ".pub", encoding="utf-8") as f:
        pub = f.read().strip()
    path = os.path.join(directory, "allowed_signers")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'{identity} namespaces="{namespace}" {pub}\n')
    return path


class TestEvidenceWorkbench(unittest.TestCase):
    """Test suite for DNA-Lang Evidence Workbench."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.capsule_path = os.path.join(self.tmp_dir.name, "test_exp.evidence.zip")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_canonical_spec_digest(self):
        """Verify the canonical specification digest is deterministic."""
        spec1 = ExperimentSpec(
            experiment_id="EXP-001",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=10,
        )
        spec2 = ExperimentSpec(
            experiment_id="EXP-001",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=10,
        )
        self.assertEqual(spec1.canonical_digest(), spec2.canonical_digest())
        self.assertEqual(len(spec1.canonical_digest()), 64)

    def test_engine_execution_and_replay(self):
        """Verify execution produces valid capsule and exact-match replay."""
        spec = ExperimentSpec(
            experiment_id="EXP-TEST-REPLAY",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=5,
            random_seed=123,
        )
        engine = EvidenceWorkbenchEngine()
        self.addCleanup(engine.ledger.close)
        result = engine.execute(spec)

        self.assertEqual(result.metrics["queries_evaluated"], 5)
        self.assertTrue(0.0 <= result.metrics["NDCG@10"] <= 1.0)
        self.assertTrue(0.0 <= result.metrics["MRR@10"] <= 1.0)
        self.assertEqual(len(result.chain_digest), 64)

        # Pack into capsule
        EvidenceCapsule.pack(result, self.capsule_path)
        self.assertTrue(os.path.exists(self.capsule_path))

        # Replay capsule
        verdict = ReplayEngine.replay(self.capsule_path)
        self.assertEqual(verdict.status, ReplayStatus.EXACT_MATCH)
        self.assertTrue(verdict.is_passing())
        self.assertTrue(verdict.chain_digest_verified)
        self.assertTrue(verdict.records_verified)
        self.assertEqual(verdict.metric_diffs["NDCG@10"], 0.0)
        self.assertEqual(verdict.metric_diffs["MRR@10"], 0.0)

    @unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
    def test_governance_authorization_gate_sys021(self):
        """Verify remote dispatch without a grant signed by an allowed signer is denied fail-closed (SYS-021)."""
        spec = ExperimentSpec(
            experiment_id="EXP-REMOTE-UNAUTHORIZED",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=5,
            execution_policy=ExecutionPolicy(mode="REMOTE_DISPATCH", require_authorization_grant=True),
        )
        key = _ssh_key(self.tmp_dir.name, "operator")
        allowed = _allowed_signers(self.tmp_dir.name, key, "devin", GRANT_NAMESPACE)
        engine = EvidenceWorkbenchEngine(allowed_signers=allowed)
        self.addCleanup(engine.ledger.close)

        with self.assertRaises(GovernanceAuthorizationViolation):
            engine.execute(spec)

        # Now test with a grant signed by the operator's key
        spec.execution_policy.grant = AuthorizationGrant.issue(
            key,
            grant_id="GRANT-2026-001",
            operator="devin",
            target_payload_sha256=spec.canonical_digest(),
            max_query_budget=100,
            expires_at_iso="2026-12-31T00:00:00Z",
        )
        # Should now succeed without raising
        result = engine.execute(spec)
        self.assertEqual(result.metrics["queries_evaluated"], 5)

    def test_quota_ceiling_gate(self):
        """Verify queries exceeding ceiling are rejected fail-closed."""
        spec = ExperimentSpec(
            experiment_id="EXP-OVER-QUOTA",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=200,
            execution_policy=ExecutionPolicy(max_query_ceiling=50),
        )
        engine = EvidenceWorkbenchEngine()
        self.addCleanup(engine.ledger.close)
        with self.assertRaises(QuotaExceededViolation):
            engine.execute(spec)

    def test_capsule_tamper_detection(self):
        """Verify that editing a capsule file without updating its checksum is caught."""
        spec = ExperimentSpec(
            experiment_id="EXP-TAMPER-TEST",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=3,
        )
        engine = EvidenceWorkbenchEngine()
        self.addCleanup(engine.ledger.close)
        result = engine.execute(spec)
        EvidenceCapsule.pack(result, self.capsule_path)

        # Mutate raw_queries.json inside the zip
        with zipfile.ZipFile(self.capsule_path, "r") as zf:
            manifest_bytes = zf.read("manifest.json")
            raw_bytes = zf.read("raw_queries.json")
            summary_bytes = zf.read("summary.md")

        mutated_raw = raw_bytes.replace(b"q-scifact", b"q-tampered")

        tampered_path = os.path.join(self.tmp_dir.name, "tampered.evidence.zip")
        with zipfile.ZipFile(tampered_path, "w") as zf:
            zf.writestr("manifest.json", manifest_bytes)
            zf.writestr("raw_queries.json", mutated_raw)
            zf.writestr("summary.md", summary_bytes)

        verdict = ReplayEngine.replay(tampered_path)
        self.assertEqual(verdict.status, ReplayStatus.TAMPER_DETECTED)
        self.assertFalse(verdict.is_passing())
        self.assertIn("Tamper detected", verdict.details)

    def test_multicorpus_execution_and_replay(self):
        """Verify multi-corpus evaluation across NFCorpus and CQADupStack."""
        for ds in ["NFCorpus", "CQADupStack"]:
            spec = ExperimentSpec(
                experiment_id=f"EXP-TEST-{ds.upper()}",
                model_id="allenai/Olmo-3-7B-Instruct",
                benchmark_suite="MTEB/BEIR",
                dataset=ds,
                split="test",
                query_count=4,
                random_seed=77,
            )
            engine = EvidenceWorkbenchEngine()
            result = engine.execute(spec)

            self.assertEqual(result.metrics["queries_evaluated"], 4)
            self.assertTrue(0.0 <= result.metrics["NDCG@10"] <= 1.0)
            self.assertTrue(0.0 <= result.metrics["MRR@10"] <= 1.0)

            capsule_p = os.path.join(self.tmp_dir.name, f"{ds}.evidence.zip")
            EvidenceCapsule.pack(result, capsule_p)
            verdict = ReplayEngine.replay(capsule_p)
            self.assertEqual(verdict.status, ReplayStatus.EXACT_MATCH)
            self.assertTrue(verdict.chain_digest_verified)

    def _spec(self, **overrides):
        fields = dict(
            experiment_id="EXP-HONESTY",
            model_id="allenai/Olmo-3-7B-Instruct",
            benchmark_suite="MTEB/BEIR",
            dataset="SciFact",
            split="test",
            query_count=5,
        )
        fields.update(overrides)
        return ExperimentSpec(**fields)

    @staticmethod
    def _read_zip(path):
        with zipfile.ZipFile(path, "r") as zf:
            return {name: zf.read(name) for name in zf.namelist()}

    @staticmethod
    def _write_zip(path, files):
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)

    def test_simulated_ndcg_never_exceeds_one(self):
        """A filler doc ID that collides with the planted relevant doc is counted once."""
        # Seed 285 produces such a collision for q-scifact-003.
        result = _execute(self._spec(query_count=10, random_seed=285))
        self.assertEqual(result.evaluated_queries[2].retrieved_doc_ids.count("doc-scifact-102"), 2,
                         "test premise: seed 285 must still produce the collision")
        for q in result.evaluated_queries:
            self.assertLessEqual(q.ndcg_at_10, 1.0, q.query_id)
            self.assertLessEqual(sum(q.relevance_scores), 1.0, q.query_id)

    def test_capsule_records_that_retrieval_is_simulated(self):
        result = _execute(self._spec())
        EvidenceCapsule.pack(result, self.capsule_path)
        data = EvidenceCapsule.unpack(self.capsule_path)
        self.assertEqual(data["manifest"]["retrieval_backend"], "simulated-seeded-rng")
        self.assertIn("Simulated retrieval", data["manifest"]["notice"])
        self.assertIn("Simulated retrieval", data["summary_md"])
        self.assertNotIn("merkle", json.dumps(data["manifest"]).lower())

    def test_rehashed_forgery_passes_checksums_but_fails_replay(self):
        """Editing a record and rewriting its checksum fools verify, not replay."""
        result = _execute(self._spec())
        EvidenceCapsule.pack(result, self.capsule_path)
        files = self._read_zip(self.capsule_path)

        records = json.loads(files["raw_queries.json"])
        records[0]["ndcg_at_10"] = 0.0 if records[0]["ndcg_at_10"] == 1.0 else 1.0
        forged = canonical_json_bytes(records)
        manifest = json.loads(files["manifest.json"])
        manifest["manifest_files"]["raw_queries.json"] = {"sha256": compute_sha256(forged), "bytes": len(forged)}
        files.update({"raw_queries.json": forged, "manifest.json": canonical_json_bytes(manifest)})
        forged_path = os.path.join(self.tmp_dir.name, "forged.evidence.zip")
        self._write_zip(forged_path, files)

        EvidenceCapsule.unpack(forged_path)  # the unsigned manifest accepts the rewritten checksum
        verdict = ReplayEngine.replay(forged_path)
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertFalse(verdict.records_verified)
        self.assertIn("per-query records differ", verdict.details)

    def _rewrite(self, edit, name="edited.evidence.zip"):
        """Pack a fresh capsule, let ``edit(files, manifest)`` change it, and write the result."""
        result = _execute(self._spec())
        EvidenceCapsule.pack(result, self.capsule_path)
        files = self._read_zip(self.capsule_path)
        manifest = json.loads(files["manifest.json"])
        edit(files, manifest)
        files["manifest.json"] = canonical_json_bytes(manifest)
        path = os.path.join(self.tmp_dir.name, name)
        self._write_zip(path, files)
        return path

    @staticmethod
    def _set_file(files, manifest, name, data):
        """Replace a capsule file and rewrite its checksum, as a forger would."""
        files[name] = data
        manifest["manifest_files"][name] = {"sha256": compute_sha256(data), "bytes": len(data)}

    def _edit_summary(self, files, manifest, fn):
        summary = fn(files["summary.md"].decode("utf-8"))
        self._set_file(files, manifest, "summary.md", summary.encode("utf-8"))

    def _replay_after(self, edit):
        return ReplayEngine.replay(self._rewrite(edit))

    def test_forged_chain_digest_alone_fails_replay(self):
        def edit(files, manifest):
            manifest["chain_digest"] = "0" * 64
        verdict = self._replay_after(edit)
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertIn("chain digest differs", verdict.details)

    def test_each_manifest_field_is_compared_with_the_replay(self):
        def queries(files, manifest):
            manifest["metrics"]["queries_evaluated"] = 5000

        def blocks(files, manifest):
            manifest["ledger_blocks_count"] = 99

        def notice(files, manifest):
            manifest["notice"] = ""

        def experiment(files, manifest):
            manifest["experiment_id"] = "EXP-OLMO3-REAL-RUN-REPRODUCED"

        for edit, message in [
            (queries, "manifest queries_evaluated 5000"),
            (blocks, "manifest ledger_blocks_count 99"),
            (notice, "manifest is missing the simulated-retrieval notice"),
            (experiment, "manifest experiment_id 'EXP-OLMO3-REAL-RUN-REPRODUCED'"),
        ]:
            verdict = self._replay_after(edit)
            self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED, edit.__name__)
            self.assertIn(message, verdict.details, edit.__name__)
            self.assertFalse(verdict.summary_verified, edit.__name__)

    def test_each_summary_figure_is_compared_with_the_replay(self):
        result = _execute(self._spec())
        cases = [
            ("mrr", f"**MRR@10:** {result.metrics['MRR@10']}", "**MRR@10:** 0.9", "summary.md MRR@10 0.9"),
            ("chain", f"`{result.chain_digest}`", "`" + "a" * 64 + "`", "summary.md chain_digest"),
            ("spec", f"`{result.spec_digest}`", "`" + "b" * 64 + "`", "summary.md spec_digest"),
            ("model", "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-7B-Instruct-REAL", "summary.md model_id"),
            ("title", "Capsule: EXP-HONESTY", "Capsule: EXP-REAL", "summary.md experiment_id"),
        ]
        for name, old, new, message in cases:
            def edit(files, manifest, old=old, new=new):
                self._edit_summary(files, manifest, lambda text: text.replace(old, new, 1))
            verdict = self._replay_after(edit)
            self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED, name)
            self.assertIn(message, verdict.details, name)

    def test_summary_notice_alone_is_checked(self):
        def edit(files, manifest):
            self._edit_summary(files, manifest, lambda text: "\n".join(
                line for line in text.splitlines() if not line.startswith("> ")))
        verdict = self._replay_after(edit)
        self.assertIn("summary.md is missing the simulated-retrieval notice", verdict.details)
        self.assertNotIn("manifest is missing", verdict.details)

    def test_missing_or_conflicting_summary_lines_are_reported(self):
        def drop_line(files, manifest):
            self._edit_summary(files, manifest, lambda text: "\n".join(
                line for line in text.splitlines() if not line.startswith("- **MRR@10:**")))

        def duplicate_line(files, manifest):
            self._edit_summary(files, manifest, lambda text: text + "- **NDCG@10:** 0.9912\n")

        self.assertIn("summary.md has no MRR@10", self._replay_after(drop_line).details)
        self.assertIn("conflicting duplicate lines", self._replay_after(duplicate_line).details)

    def test_missing_backend_label_is_tamper(self):
        def edit(files, manifest):
            del manifest["retrieval_backend"]
        verdict = self._replay_after(edit)
        self.assertEqual(verdict.status, ReplayStatus.TAMPER_DETECTED)
        self.assertIn("no retrieval_backend label", verdict.details)

    def test_truncated_records_with_rewritten_checksum_fail_replay(self):
        def edit(files, manifest):
            records = json.loads(files["raw_queries.json"])[:-1]
            self._set_file(files, manifest, "raw_queries.json", canonical_json_bytes(records))
        verdict = self._replay_after(edit)
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertIn("1 of 5 per-query records differ", verdict.details)

    def test_policy_refusal_is_not_replayable(self):
        def edit(files, manifest):
            # Outside the canonical digest, so only the policy gate can object.
            manifest["spec"]["execution_policy"]["require_authorization_grant"] = True
        verdict = self._replay_after(edit)
        self.assertEqual(verdict.status, ReplayStatus.NOT_REPLAYABLE)
        self.assertIn("policy gate refused", verdict.details)

    def test_spec_digest_mismatch_is_tamper(self):
        def edit(files, manifest):
            manifest["spec"]["random_seed"] = 43
        verdict = self._replay_after(edit)
        self.assertEqual(verdict.status, ReplayStatus.TAMPER_DETECTED)
        self.assertIn("Canonical spec digest mismatch", verdict.details)

    def test_forged_chain_digest_fails_replay_even_with_metrics_nudged(self):
        def edit(files, manifest):
            manifest["chain_digest"] = "f" * 64
            manifest["metrics"]["CTC"] += 5e-5
        verdict = ReplayEngine.replay(self._rewrite(edit))
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertFalse(verdict.chain_digest_verified)

    def test_simulated_capsule_metrics_must_match_exactly(self):
        def edit(files, manifest):
            manifest["metrics"]["NDCG@10"] += 9e-5  # within the default 1e-4 tolerance
        verdict = ReplayEngine.replay(self._rewrite(edit))
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertIn("metric drift", verdict.details)

    def test_unpack_rejects_duplicate_zip_entries(self):
        result = _execute(self._spec())
        EvidenceCapsule.pack(result, self.capsule_path)
        files = self._read_zip(self.capsule_path)
        path = os.path.join(self.tmp_dir.name, "dup.evidence.zip")
        with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as zf:
            warnings.simplefilter("ignore", UserWarning)  # the duplicate is the point
            zf.writestr("raw_queries.json", b"[]")
            for name, data in files.items():
                zf.writestr(name, data)
        with self.assertRaises(ValueError):
            EvidenceCapsule.unpack(path)

    def test_relabeled_capsule_passes_checksums_but_fails_replay(self):
        def edit(files, manifest):
            summary = files["summary.md"].decode("utf-8")
            summary = "\n".join(l for l in summary.splitlines() if not l.startswith("> "))
            summary = summary.replace("(not loaded)", "").encode("utf-8")
            files["summary.md"] = summary
            manifest["manifest_files"]["summary.md"] = {"sha256": compute_sha256(summary), "bytes": len(summary)}
            manifest["notice"] = ""
        path = self._rewrite(edit)
        EvidenceCapsule.unpack(path)  # checksums still match
        verdict = ReplayEngine.replay(path)
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertFalse(verdict.summary_verified)
        self.assertIn("simulated-retrieval notice", verdict.details)

    def test_inflated_summary_headline_fails_replay(self):
        def edit(files, manifest):
            summary = files["summary.md"].decode("utf-8")
            stored = manifest["metrics"]["NDCG@10"]
            summary = summary.replace(f"**NDCG@10:** {stored}", "**NDCG@10:** 0.9912").encode("utf-8")
            files["summary.md"] = summary
            manifest["manifest_files"]["summary.md"] = {"sha256": compute_sha256(summary), "bytes": len(summary)}
        verdict = ReplayEngine.replay(self._rewrite(edit))
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertIn("summary.md NDCG@10 0.9912", verdict.details)

    def test_capsule_claiming_another_backend_is_not_replayable(self):
        def edit(files, manifest):
            manifest["retrieval_backend"] = "olmo3-7b-dense"
        verdict = ReplayEngine.replay(self._rewrite(edit))
        self.assertEqual(verdict.status, ReplayStatus.NOT_REPLAYABLE)
        self.assertFalse(verdict.is_passing())

    @unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
    def test_authorized_remote_dispatch_capsule_replays(self):
        key = _ssh_key(self.tmp_dir.name, "operator")
        allowed = _allowed_signers(self.tmp_dir.name, key, "devin", GRANT_NAMESPACE)
        spec = self._spec(execution_policy=ExecutionPolicy(mode="REMOTE_DISPATCH", require_authorization_grant=True))
        spec.execution_policy.grant = AuthorizationGrant.issue(
            key, grant_id="GRANT-TEST", operator="devin", target_payload_sha256=spec.canonical_digest(),
            max_query_budget=100, expires_at_iso="2099-01-01T00:00:00Z",
        )
        engine = EvidenceWorkbenchEngine(allowed_signers=allowed)
        try:
            result = engine.execute(spec)
        finally:
            engine.ledger.close()
        EvidenceCapsule.pack(result, self.capsule_path)
        self.assertEqual(ReplayEngine.replay(self.capsule_path, allowed_signers=allowed).status,
                         ReplayStatus.EXACT_MATCH)
        # Without the allowed signers the stored grant cannot be checked, so the re-run is refused
        self.assertEqual(ReplayEngine.replay(self.capsule_path).status, ReplayStatus.NOT_REPLAYABLE)

    def test_unpack_rejects_incomplete_or_padded_capsules(self):
        def drop_digest(files, manifest):
            del manifest["chain_digest"]

        def drop_summary_checksum(files, manifest):
            del manifest["manifest_files"]["summary.md"]

        def add_file(files, manifest):
            files["extra.txt"] = b"unlisted"

        def records_not_a_list(files, manifest):
            files["raw_queries.json"] = b"5"
            manifest["manifest_files"]["raw_queries.json"] = {"sha256": compute_sha256(b"5"), "bytes": 1}

        def metrics_not_an_object(files, manifest):
            manifest["metrics"] = [0.5]

        for edit, message in [
            (drop_digest, "manifest is missing chain_digest"),
            (drop_summary_checksum, "manifest has no checksum for 'summary.md'"),
            (add_file, "files not listed in the manifest: extra.txt"),
            (records_not_a_list, "raw_queries.json is not a list of query records"),
            (metrics_not_an_object, "manifest 'metrics' is not a JSON object"),
        ]:
            with self.assertRaises(ValueError, msg=edit.__name__) as caught:
                EvidenceCapsule.unpack(self._rewrite(edit, f"{edit.__name__}.evidence.zip"))
            self.assertIn(message, str(caught.exception), edit.__name__)

    def test_unpack_rejects_a_manifest_that_is_not_an_object(self):
        result = _execute(self._spec())
        EvidenceCapsule.pack(result, self.capsule_path)
        files = self._read_zip(self.capsule_path)
        for bad in (b"5", b"null", b'["merkle_root"]', b"[]"):
            files["manifest.json"] = bad
            path = os.path.join(self.tmp_dir.name, "bad_manifest.evidence.zip")
            self._write_zip(path, files)
            with self.assertRaises(ValueError, msg=bad) as caught:
                EvidenceCapsule.unpack(path)
            self.assertIn("not a JSON object", str(caught.exception), bad)

    def test_metrics_at_10_ignore_ranks_beyond_10(self):
        result = _execute(self._spec(top_k=20, query_count=10, random_seed=7))
        for q in result.evaluated_queries:
            self.assertEqual(len(q.relevance_scores), 20)
            self.assertEqual(sum(q.relevance_scores[10:]), 0.0, q.query_id)
            if q.ndcg_at_10 == 0.0:
                self.assertEqual(sum(q.relevance_scores), 0.0, q.query_id)

    def test_v1_0_0_capsule_with_merkle_root_key_still_replays(self):
        result = _execute(self._spec())
        EvidenceCapsule.pack(result, self.capsule_path)
        files = self._read_zip(self.capsule_path)
        manifest = json.loads(files["manifest.json"])
        manifest["merkle_root"] = manifest.pop("chain_digest")
        manifest["version"] = "1.0.0"
        for key in ("retrieval_backend", "notice"):
            manifest.pop(key)
        files["manifest.json"] = canonical_json_bytes(manifest)
        legacy_path = os.path.join(self.tmp_dir.name, "legacy.evidence.zip")
        self._write_zip(legacy_path, files)

        self.assertEqual(EvidenceCapsule.unpack(legacy_path)["manifest"]["chain_digest"], result.chain_digest)
        self.assertEqual(ReplayEngine.replay(legacy_path).status, ReplayStatus.EXACT_MATCH)


if __name__ == "__main__":
    unittest.main()

