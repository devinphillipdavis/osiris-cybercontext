"""Tests for the BM25 analyzers (workbench/analysis.py) and the analyzer option of the real run."""

import json
import os
import tempfile
import unittest

from workbench.analysis import ANALYZERS, LUCENE_ENGLISH, SIMPLE, analyze, lucene_length_norm, porter_stem
from workbench.capsule import EvidenceCapsule
from workbench.replay import ReplayEngine, ReplayStatus
from workbench.real_scifact import BM25Index, REAL_BACKEND, load_beir_dataset, run_real_scifact, tokenize
from workbench.schema import ExecutionPolicy, ExperimentSpec

# Porter's published sample pairs (word, stem) from the 1980 paper and its test vocabulary.
PORTER_PAIRS = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "plastered": "plaster", "bled": "bled", "motoring": "motor", "sing": "sing",
    "conflated": "conflat", "troubled": "troubl", "sized": "size", "hopping": "hop",
    "tanned": "tan", "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail",
    "filing": "file", "happy": "happi", "sky": "sky", "relational": "relat",
    "conditional": "condit", "rational": "ration", "valenci": "valenc", "hesitanci": "hesit",
    "digitizer": "digit", "conformabli": "conform", "radicalli": "radic",
    "differentli": "differ", "vileli": "vile", "analogousli": "analog",
    "vietnamization": "vietnam", "predication": "predic", "operator": "oper",
    "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
    "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
    "sensibiliti": "sensibl", "triplicate": "triplic", "formative": "form",
    "formalize": "formal", "electriciti": "electr", "electrical": "electr", "hopeful": "hope",
    "goodness": "good", "revival": "reviv", "allowance": "allow", "inference": "infer",
    "airliner": "airlin", "gyroscopic": "gyroscop", "adjustable": "adjust",
    "defensible": "defens", "irritant": "irrit", "replacement": "replac",
    "adjustment": "adjust", "dependent": "depend", "adoption": "adopt", "communism": "commun",
    "activate": "activ", "angulariti": "angular", "homologous": "homolog",
    "effective": "effect", "bowdlerize": "bowdler", "probate": "probat", "rate": "rate",
    "cease": "ceas", "controll": "control", "roll": "roll",
}


class TestPorterStemmer(unittest.TestCase):
    def test_published_sample_pairs(self):
        for word, stem in PORTER_PAIRS.items():
            with self.subTest(word=word):
                self.assertEqual(porter_stem(word), stem)

    def test_short_words_are_left_alone(self):
        self.assertEqual(porter_stem("is"), "is")
        self.assertEqual(porter_stem("a"), "a")


class TestAnalyzers(unittest.TestCase):
    def test_simple_analyzer_is_the_original_tokenizer(self):
        text = "Hello, World-2! NDCG@10 (test) don't"
        self.assertEqual(analyze(text, SIMPLE), tokenize(text))
        self.assertEqual(analyze(text), ["hello", "world", "2", "ndcg", "10", "test", "don", "t"])

    def test_lucene_english_keeps_apostrophes_periods_and_digit_groups(self):
        self.assertIn("don't", analyze("don't panic", LUCENE_ENGLISH))
        self.assertIn("0.05", analyze("p < 0.05", LUCENE_ENGLISH))
        self.assertIn("1,000", analyze("in 1,000 trials", LUCENE_ENGLISH))

    def test_lucene_english_splits_hyphens_and_strips_possessives(self):
        tokens = analyze("COVID-19 and John's data", LUCENE_ENGLISH)
        self.assertIn("covid", tokens)
        self.assertIn("19", tokens)
        self.assertIn("john", tokens)
        self.assertNotIn("john's", tokens)

    def test_lucene_english_drops_stopwords_and_stems(self):
        tokens = analyze("The runners were running in the lab", LUCENE_ENGLISH)
        self.assertEqual(tokens, ["runner", "were", "run", "lab"])  # "were" is not on Lucene's stop list

    def test_unknown_analyzer_is_refused(self):
        with self.assertRaises(ValueError):
            analyze("text", "whitespace")
        self.assertEqual(set(ANALYZERS), {SIMPLE, LUCENE_ENGLISH})

    def test_lucene_tokenizer_splits_where_uax29_splits(self):
        # possessive after a digit: "cd33" and a lone "s"; a period between letter and digit splits
        self.assertEqual(analyze("CD33\u2019s", LUCENE_ENGLISH), ["cd33", "s"])
        self.assertEqual(analyze("aging.1", LUCENE_ENGLISH), ["ag", "1"])  # "aging" stems to "ag"
        self.assertEqual(analyze("km\u00b2 and nh\u2082", LUCENE_ENGLISH), ["km", "nh"])
        self.assertEqual(analyze("___ x", LUCENE_ENGLISH), ["x"])

    def test_lucene_tokenizer_keeps_emoji_symbols_as_tokens(self):
        self.assertEqual(analyze("Pfizer\u00ae trial", LUCENE_ENGLISH), ["pfizer", "\u00ae", "trial"])

    def test_format_characters_attach_to_the_letter_before_them(self):
        self.assertEqual(analyze("shrink\u00ad age", LUCENE_ENGLISH), ["shrink\u00ad", "ag"])
        self.assertEqual(analyze("\u200bcells", LUCENE_ENGLISH), ["cell"])


class TestLuceneLengthEncoding(unittest.TestCase):
    def test_lengths_below_24_are_exact(self):
        for n in range(24):
            self.assertEqual(lucene_length_norm(n), n)

    def test_longer_lengths_keep_four_significant_bits(self):
        self.assertEqual(lucene_length_norm(32), 32)  # representable
        self.assertEqual(lucene_length_norm(40), 40)  # representable
        self.assertEqual(lucene_length_norm(41), 40)  # rounds down
        self.assertEqual(lucene_length_norm(100), 96)
        for n in range(24, 2000):
            rounded = lucene_length_norm(n)
            self.assertLessEqual(rounded, n)
            self.assertGreater(rounded, n * 0.85)


class TestLuceneScoring(unittest.TestCase):
    def test_repeated_query_terms_are_scored_per_occurrence(self):
        corpus = {"d1": ["tumor", "growth", "cells"], "d2": ["cells", "cells", "growth", "tumor"]}
        lucene = BM25Index({d: analyze(" ".join(t), LUCENE_ENGLISH) for d, t in corpus.items()}, analyzer=LUCENE_ENGLISH)
        once = dict(lucene.search("tumor", top_k=2))
        twice = dict(lucene.search("tumor tumor", top_k=2))
        self.assertAlmostEqual(twice["d1"], 2 * once["d1"])
        simple = BM25Index(corpus)
        self.assertAlmostEqual(dict(simple.search("tumor tumor", top_k=2))["d1"],
                               dict(simple.search("tumor", top_k=2))["d1"])


TOY_CORPUS = {
    "d1": "The runners were running quickly through the park.",
    "d2": "Cats sleep for most of the day.",
    "d3": "Quantum error correction codes protect logical qubits.",
}
TOY_QUERIES = {"q1": "run"}
TOY_QRELS = {"q1": {"d1": 1}}


def _write_toy_dataset(root):
    data_dir = os.path.join(root, "scifact")
    os.makedirs(os.path.join(data_dir, "qrels"), exist_ok=True)
    with open(os.path.join(data_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, text in TOY_CORPUS.items():
            f.write(json.dumps({"_id": doc_id, "title": "", "text": text}) + "\n")
    with open(os.path.join(data_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for query_id, text in TOY_QUERIES.items():
            f.write(json.dumps({"_id": query_id, "text": text}) + "\n")
    with open(os.path.join(data_dir, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, docs in TOY_QRELS.items():
            for doc_id, grade in docs.items():
                f.write(f"{query_id}\t{doc_id}\t{grade}\n")
    return data_dir


def _spec(analyzer=None):
    options = {"analyzer": analyzer} if analyzer else {}
    return ExperimentSpec(
        experiment_id="EXP-TOY-ANALYZER-001", model_id="bm25-stdlib", benchmark_suite="MTEB/BEIR",
        dataset="SciFact", split="test", query_count=1, top_k=10, similarity_metric="bm25",
        random_seed=42, retrieval_options=options,
        execution_policy=ExecutionPolicy(mode="LOCAL_BOUNDED", max_query_ceiling=10),
    )


class TestAnalyzerInTheRealRun(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.data_dir = _write_toy_dataset(self.dir)

    def test_stemming_lets_a_query_match_its_inflected_forms(self):
        lucene = run_real_scifact(_spec(LUCENE_ENGLISH), self.data_dir, ledger=None)
        simple = run_real_scifact(_spec(None), self.data_dir, ledger=None)
        self.assertEqual(lucene.metrics["NDCG@10"], 1.0)
        self.assertEqual(simple.metrics["NDCG@10"], 0.0)
        self.assertEqual(lucene.retrieval_backend, REAL_BACKEND)

    def test_the_analyzer_is_part_of_the_spec_identity(self):
        self.assertNotEqual(_spec(LUCENE_ENGLISH).canonical_digest(), _spec(None).canonical_digest())

    def test_a_lucene_english_capsule_replays_with_its_analyzer(self):
        result = run_real_scifact(_spec(LUCENE_ENGLISH), self.data_dir, ledger=None)
        capsule = EvidenceCapsule.pack(result, os.path.join(self.dir, "lucene.evidence.zip"))
        verdict = ReplayEngine.replay(capsule, real_data_dir=self.data_dir)
        self.assertEqual(verdict.status, ReplayStatus.EXACT_MATCH, verdict.details)

    def test_unknown_analyzer_and_dense_analyzer_are_refused(self):
        with self.assertRaises(ValueError):
            run_real_scifact(_spec("whitespace"), self.data_dir, ledger=None)
        dense = _spec(LUCENE_ENGLISH)
        dense.retrieval_options["backend"] = "dense"
        with self.assertRaises(ValueError):
            run_real_scifact(dense, self.data_dir, ledger=None)


@unittest.skipUnless(os.environ.get("OSIRIS_SCIFACT_DIR"), "set OSIRIS_SCIFACT_DIR to a BEIR SciFact directory")
class TestLuceneParityOnSciFact(unittest.TestCase):
    """The lucene-english analyzer reproduces Pyserini's flat BM25 index on the SciFact test split.

    Checked against the prebuilt beir-v1.0.0-scifact.flat index (k1=0.9, b=0.4): identical top-10
    rankings for all 300 judged queries, mean NDCG@10 0.6789 and mean MRR@10 0.6457.
    """

    def test_mean_metrics_match_pyserini_flat_index(self):
        spec = ExperimentSpec(
            experiment_id="EXP-SCIFACT-PARITY", model_id="bm25-stdlib", benchmark_suite="MTEB/BEIR",
            dataset="SciFact", split="test", query_count=len(load_beir_dataset(os.environ["OSIRIS_SCIFACT_DIR"]).queries),
            top_k=10, similarity_metric="bm25", random_seed=42,
            retrieval_options={"analyzer": LUCENE_ENGLISH},
            execution_policy=ExecutionPolicy(mode="LOCAL_BOUNDED", max_query_ceiling=1000),
        )
        result = run_real_scifact(spec, os.environ["OSIRIS_SCIFACT_DIR"])
        self.assertEqual(round(result.metrics["NDCG@10"], 4), 0.6789)
        self.assertEqual(round(result.metrics["MRR@10"], 4), 0.6457)


if __name__ == "__main__":
    unittest.main()
