"""Tests for the optional real SciFact retrieval backend (BM25 over local BEIR data).

The synthetic toy dataset below is written into a temp directory, so no
network access or dataset download is needed. The expected rankings and metric
values were derived with an independent port of the BM25 and graded-NDCG logic
and are asserted exactly, so a change to the tokenizer, the BM25 parameters or
the metric formulas shows up here.

Set ``OSIRIS_SCIFACT_DIR`` to a local BEIR SciFact directory to also run the
last test against the real corpus.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

from core.ast_firewall import CyberContextASTFirewall
from workbench.capsule import EvidenceCapsule
from workbench.engine import QuotaExceededViolation
from workbench.replay import ReplayEngine, ReplayStatus
from workbench.real_scifact import (
    BM25_B,
    BM25_K1,
    REAL_BACKEND,
    graded_ndcg_at_10,
    load_beir_dataset,
    mrr_at_10,
    run_real_scifact,
    tokenize,
)
from workbench.schema import ExperimentSpec, ExecutionPolicy, canonical_json_bytes

# 4 documents / 3 queries / graded qrels, designed so BM25 separates them cleanly.
TOY_CORPUS = {
    "doc-1": "alpha gamma common",
    "doc-2": "delta gamma common",
    "doc-3": "delta gamma unique three",
    "doc-4": "alpha alpha unique four",
}
TOY_QUERIES = {"q-1": "alpha", "q-2": "gamma", "q-3": "delta"}
# Listed in queries.jsonl but judged only in another split, as SciFact's train queries are.
# It sorts first, so it would displace q-2 from a --queries 2 run if it were not filtered out.
UNJUDGED_QUERIES = {"q-0": "alpha gamma"}
TOY_QRELS = {
    "q-1": {"doc-4": 2, "doc-1": 1},
    "q-2": {"doc-3": 2},
    "q-3": {"doc-3": 2, "doc-2": 1},
}

# Expected per-query results over the toy corpus (BM25, k1=0.9, b=0.4),
# verified with an independent implementation of the same math.
TOY_EXPECTED = {
    "q-1": {"retrieved": ["doc-4", "doc-1"], "ndcg": 1.0, "mrr": 1.0, "grades": [2.0, 1.0]},
    "q-2": {"retrieved": ["doc-1", "doc-2", "doc-3"], "ndcg": 0.5, "mrr": 0.3333, "grades": [0.0, 0.0, 2.0]},
    "q-3": {"retrieved": ["doc-2", "doc-3"], "ndcg": 0.7967, "mrr": 1.0, "grades": [1.0, 2.0]},
}
TOY_MEAN_NDCG = 0.7656
TOY_MEAN_MRR = 0.7778
TOY_CTC = 0.232


def _write_toy_dataset(root: str) -> str:
    """Write a BEIR-layout SciFact-shaped toy dataset under ``root`` and return its path."""
    data_dir = os.path.join(root, "scifact")
    os.makedirs(os.path.join(data_dir, "qrels"), exist_ok=True)
    with open(os.path.join(data_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, text in TOY_CORPUS.items():
            f.write(json.dumps({"_id": doc_id, "title": "", "text": text}) + "\n")
    with open(os.path.join(data_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for query_id, text in {**UNJUDGED_QUERIES, **TOY_QUERIES}.items():
            f.write(json.dumps({"_id": query_id, "text": text}) + "\n")
    with open(os.path.join(data_dir, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, docs in sorted(TOY_QRELS.items()):
            for doc_id, grade in sorted(docs.items()):
                f.write(f"{query_id}\t{doc_id}\t{grade}\n")
    return data_dir


def _toy_spec(**overrides) -> ExperimentSpec:
    fields = dict(
        experiment_id="EXP-TOY-REAL-001",
        model_id="bm25-stdlib",
        benchmark_suite="MTEB/BEIR",
        dataset="SciFact",
        split="test",
        query_count=3,
        top_k=10,
        similarity_metric="bm25",
        random_seed=42,
        execution_policy=ExecutionPolicy(mode="LOCAL_BOUNDED", max_query_ceiling=10),
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


class TestRealScifactBackend(unittest.TestCase):
    """BM25 backend over a synthetic BEIR-layout toy dataset."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.data_dir = _write_toy_dataset(self.tmp_dir.name)
        self.capsule_path = os.path.join(self.tmp_dir.name, "toy_real.evidence.zip")

    def test_tokenizer_is_a_simple_lowercase_alnum_split(self):
        self.assertEqual(tokenize("Hello, World-2!"), ["hello", "world", "2"])
        self.assertEqual(tokenize("NDCG@10 (test)"), ["ndcg", "10", "test"])
        self.assertEqual(tokenize(""), [])

    def test_graded_ndcg_hand_case(self):
        # grades [2, 0, 1, 0] -> DCG 3 + 1/2 = 3.5; ideal [2, 1] -> IDCG 3 + 1/log2(3).
        value = graded_ndcg_at_10(["doc-1", "doc-2", "doc-3", "doc-4"], {"doc-1": 2, "doc-3": 1})
        self.assertAlmostEqual(value, 0.9639, places=4)
        self.assertEqual(graded_ndcg_at_10(["doc-x"], {"doc-y": 2}), 0.0)

    def test_mrr_at_10(self):
        self.assertEqual(mrr_at_10(["doc-2", "doc-3"], {"doc-3": 2}), 0.5)
        self.assertEqual(mrr_at_10(["doc-2"], {"doc-3": 2}), 0.0)

    def test_toy_run_rankings_and_metrics(self):
        """The BM25 rankings and graded metrics over the toy corpus match the reference values."""
        result = run_real_scifact(_toy_spec(), self.data_dir)

        self.assertEqual(result.retrieval_backend, REAL_BACKEND)
        self.assertEqual(result.metrics["queries_evaluated"], 3)
        self.assertEqual(result.metrics["NDCG@10"], TOY_MEAN_NDCG)
        self.assertEqual(result.metrics["MRR@10"], TOY_MEAN_MRR)
        self.assertEqual(result.metrics["CTC"], TOY_CTC)
        self.assertEqual([q.query_id for q in result.evaluated_queries], ["q-1", "q-2", "q-3"])

        for record in result.evaluated_queries:
            expected = TOY_EXPECTED[record.query_id]
            self.assertEqual(record.retrieved_doc_ids, expected["retrieved"], record.query_id)
            self.assertEqual(record.ndcg_at_10, expected["ndcg"], record.query_id)
            self.assertEqual(record.mrr_at_10, expected["mrr"], record.query_id)
            # relevance_scores are qrels grades, not retriever scores
            self.assertEqual(record.relevance_scores, expected["grades"], record.query_id)

        telemetry = result.environment_telemetry
        self.assertEqual(telemetry["retrieval_backend"], REAL_BACKEND)
        self.assertEqual(telemetry["bm25_k1"], BM25_K1)
        self.assertEqual(telemetry["bm25_b"], BM25_B)
        self.assertEqual(telemetry["corpus_documents"], 4)
        self.assertEqual(telemetry["queries_in_split"], 3)
        self.assertEqual(telemetry["queries_in_file"], 4)

    def test_query_count_selects_the_first_queries_in_sorted_id_order(self):
        result = run_real_scifact(_toy_spec(query_count=2), self.data_dir)
        self.assertEqual([q.query_id for q in result.evaluated_queries], ["q-1", "q-2"])
        self.assertEqual(result.metrics["queries_evaluated"], 2)

    def test_queries_without_judgments_in_the_split_are_not_evaluated(self):
        data = load_beir_dataset(self.data_dir)
        self.assertEqual([q for q, _ in data.queries], ["q-1", "q-2", "q-3"])
        self.assertEqual(data.queries_in_file, 4)
        result = run_real_scifact(_toy_spec(query_count=5), self.data_dir)  # more than the 3 judged
        self.assertNotIn("q-0", [q.query_id for q in result.evaluated_queries])
        self.assertEqual(result.metrics["queries_evaluated"], 3)

    def test_two_runs_are_bit_for_bit_deterministic(self):
        first = run_real_scifact(_toy_spec(), self.data_dir)
        second = run_real_scifact(_toy_spec(), self.data_dir)
        self.assertEqual(first.chain_digest, second.chain_digest)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(
            [q.query_digest for q in first.evaluated_queries],
            [q.query_digest for q in second.evaluated_queries],
        )

    def test_capsule_roundtrip_replays_with_the_dataset_directory(self):
        result = run_real_scifact(_toy_spec(), self.data_dir)
        EvidenceCapsule.pack(result, self.capsule_path)

        data = EvidenceCapsule.unpack(self.capsule_path)
        self.assertEqual(data["manifest"]["retrieval_backend"], REAL_BACKEND)
        self.assertTrue(data["manifest"]["notice"].startswith("Real retrieval"))
        self.assertIn("Real retrieval", data["summary_md"])

        verdict = ReplayEngine.replay(self.capsule_path, real_data_dir=self.data_dir)
        self.assertEqual(verdict.status, ReplayStatus.EXACT_MATCH)
        self.assertTrue(verdict.is_passing())
        self.assertTrue(verdict.records_verified)
        self.assertTrue(verdict.chain_digest_verified)

    def test_real_capsule_without_the_dataset_directory_is_not_replayable(self):
        result = run_real_scifact(_toy_spec(), self.data_dir)
        EvidenceCapsule.pack(result, self.capsule_path)

        verdict = ReplayEngine.replay(self.capsule_path)
        self.assertEqual(verdict.status, ReplayStatus.NOT_REPLAYABLE)
        self.assertFalse(verdict.is_passing())
        self.assertIn("SciFact directory", verdict.details)

    def test_tampered_manifest_metrics_fail_replay(self):
        """Inflating the manifest's headline NDCG (manifest is unsigned) is caught by replay."""
        result = run_real_scifact(_toy_spec(), self.data_dir)
        EvidenceCapsule.pack(result, self.capsule_path)
        with zipfile.ZipFile(self.capsule_path, "r") as zf:
            files = {name: zf.read(name) for name in zf.namelist()}

        manifest = json.loads(files["manifest.json"])
        manifest["metrics"]["NDCG@10"] = 0.9912
        files["manifest.json"] = canonical_json_bytes(manifest)
        tampered_path = os.path.join(self.tmp_dir.name, "tampered.evidence.zip")
        with zipfile.ZipFile(tampered_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

        EvidenceCapsule.unpack(tampered_path)  # checksums still pass: the manifest is unsigned
        verdict = ReplayEngine.replay(tampered_path, real_data_dir=self.data_dir)
        self.assertEqual(verdict.status, ReplayStatus.DRIFT_DETECTED)
        self.assertIn("metric drift", verdict.details)

    def test_query_ceiling_gate_applies_to_the_real_backend(self):
        with self.assertRaises(QuotaExceededViolation):
            run_real_scifact(_toy_spec(query_count=20), self.data_dir)

    def test_missing_dataset_directory_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            run_real_scifact(_toy_spec(), os.path.join(self.tmp_dir.name, "no_such_dir"))

    def test_loader_rejects_rows_without_an_id(self):
        bad_dir = _write_toy_dataset(os.path.join(self.tmp_dir.name, "bad"))
        with open(os.path.join(bad_dir, "queries.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": "no id"}) + "\n")
        with self.assertRaises(ValueError):
            load_beir_dataset(bad_dir)

    def test_loader_accepts_abstract_field_and_four_column_qrels(self):
        """The original SciFact release names the text field 'abstract'; qrels may have an iteration column."""
        variant_dir = os.path.join(self.tmp_dir.name, "variant")
        os.makedirs(os.path.join(variant_dir, "qrels"), exist_ok=True)
        with open(os.path.join(variant_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for doc_id, text in TOY_CORPUS.items():
                f.write(json.dumps({"_id": doc_id, "title": "", "abstract": text}) + "\n")
        with open(os.path.join(variant_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
            for query_id, text in TOY_QUERIES.items():
                f.write(json.dumps({"_id": query_id, "text": text}) + "\n")
        with open(os.path.join(variant_dir, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
            f.write("query-id\titeration\tcorpus-id\tscore\n")
            for query_id, docs in sorted(TOY_QRELS.items()):
                for doc_id, grade in sorted(docs.items()):
                    f.write(f"{query_id}\t0\t{doc_id}\t{grade}\n")

        result = run_real_scifact(_toy_spec(), variant_dir)
        self.assertEqual(result.metrics["NDCG@10"], TOY_MEAN_NDCG)
        self.assertEqual(result.metrics["MRR@10"], TOY_MEAN_MRR)


class TestShellTrueScreening(unittest.TestCase):
    """The AST screen flags any call with shell=True, including aliased subprocess calls."""

    def test_aliased_subprocess_with_shell_true_is_flagged(self):
        turn = 'import subprocess as s\ns.run("dir", shell=True)'
        verdict = CyberContextASTFirewall.sanitize_turn(turn)
        self.assertFalse(verdict.is_safe)
        self.assertEqual(verdict.detection_layer, "ast")
        self.assertIn("shell=True", verdict.quarantine_reason)

    def test_aliased_subprocess_without_shell_true_still_passes_the_ast_layer(self):
        # The dotted-name check does not know 's.run'; this is a known limit of the screen.
        turn = 'import subprocess as s\ns.run(["ls", "-la"])'
        verdict = CyberContextASTFirewall.sanitize_turn(turn)
        self.assertTrue(verdict.is_safe, verdict.quarantine_reason)

    def test_arbitrary_dotted_call_with_shell_true_is_flagged(self):
        verdict = CyberContextASTFirewall.sanitize_turn("shell Runner: foo.bar(cmd, shell=True)")
        # 'shell Runner: ...' does not parse as Python, so the AST layer is skipped.
        self.assertTrue(verdict.is_safe)

        verdict = CyberContextASTFirewall.sanitize_turn("foo.bar(cmd, shell=True)")
        self.assertFalse(verdict.is_safe)
        self.assertEqual(verdict.detection_layer, "ast")
        self.assertIn("'foo.bar()'", verdict.quarantine_reason)

    def test_shell_false_and_shell_variable_are_not_flagged(self):
        for turn in ("foo.bar(cmd, shell=False)", "foo.bar(cmd, shell=use_shell)"):
            verdict = CyberContextASTFirewall.sanitize_turn(turn)
            self.assertTrue(verdict.is_safe, turn)


class TestRunRealCli(unittest.TestCase):
    """End-to-end test of the ``run-real`` and ``replay --data`` CLI wiring, as subprocesses."""

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.data_dir = _write_toy_dataset(self.tmp_dir.name)
        self.cwd = os.path.join(self.tmp_dir.name, "cwd")
        os.makedirs(self.cwd, exist_ok=True)

    def _run(self, *args):
        env = dict(os.environ, PYTHONPATH=self.REPO_ROOT, PYTHONIOENCODING="utf-8")
        return subprocess.run(
            [sys.executable, "-m", "workbench.cli", *args], cwd=self.cwd, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )

    def test_run_real_and_replay_round_trip(self):
        capsule = "real_capsule.evidence.zip"
        run = self._run("run-real", "--data", self.data_dir, "--out", capsule)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("Real retrieval", run.stdout)
        self.assertIn(REAL_BACKEND, run.stdout)
        self.assertIn(str(TOY_MEAN_NDCG), run.stdout)

        replay = self._run("replay", capsule, "--data", self.data_dir)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        self.assertIn("EXACT_MATCH", replay.stdout)

        without_data = self._run("replay", capsule)
        self.assertEqual(without_data.returncode, 1, without_data.stdout + without_data.stderr)
        self.assertIn("NOT_REPLAYABLE", without_data.stdout)
        self.assertIn("--data", without_data.stdout)

    def test_run_real_reports_a_missing_dataset_directory(self):
        run = self._run("run-real", "--data", os.path.join(self.tmp_dir.name, "no_such_dir"))
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("Could not load the SciFact dataset directory", run.stdout)
        self.assertEqual(os.listdir(self.cwd), [])


@unittest.skipUnless(
    os.environ.get("OSIRIS_SCIFACT_DIR"),
    "set OSIRIS_SCIFACT_DIR to a local BEIR SciFact directory to run this test",
)
class TestRealCorpus(unittest.TestCase):
    """Sanity check over the real BEIR SciFact corpus (opt-in, no network)."""

    def test_real_corpus_run_replays_exactly(self):
        data_dir = os.environ["OSIRIS_SCIFACT_DIR"]
        queries = int(os.environ.get("OSIRIS_SCIFACT_QUERIES", "25"))
        spec = _toy_spec(
            experiment_id="EXP-SCIFACT-REAL-001", query_count=queries,
            execution_policy=ExecutionPolicy(mode="LOCAL_BOUNDED", max_query_ceiling=queries),
        )
        result = run_real_scifact(spec, data_dir)

        self.assertEqual(result.retrieval_backend, REAL_BACKEND)
        self.assertEqual(result.metrics["queries_evaluated"], queries)
        self.assertTrue(0.0 <= result.metrics["NDCG@10"] <= 1.0)
        self.assertTrue(0.0 <= result.metrics["MRR@10"] <= 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            capsule_path = os.path.join(tmp, "real.evidence.zip")
            EvidenceCapsule.pack(result, capsule_path)
            verdict = ReplayEngine.replay(capsule_path, real_data_dir=data_dir)
            self.assertEqual(verdict.status, ReplayStatus.EXACT_MATCH)

            no_data = ReplayEngine.replay(capsule_path)
            self.assertEqual(no_data.status, ReplayStatus.NOT_REPLAYABLE)


if __name__ == "__main__":
    unittest.main()
