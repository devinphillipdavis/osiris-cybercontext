"""Real SciFact retrieval backends (optional extra): BM25, dense embeddings, LLM reranking.

This module adds real retrieval runs to the evidence workbench over a local
copy of a BEIR dataset (SciFact by default), scored with graded NDCG@10 /
MRR@10 against the dataset's qrels and recorded in the same hash-chained
ledger and capsule format as the simulated engine. The backend comes from
``spec.retrieval_options["backend"]``:

- ``bm25`` (default, standard library only): an Okapi BM25 index.
- ``dense``: a sentence-transformers embedding model named by ``spec.model_id``
  (optional extras, see requirements-real.txt), optionally followed by an
  experimental LLM reranker (``retrieval_options["rerank_model"]``, for
  example allenai/Olmo-3-7B-Instruct). See workbench/real_models.py.

BM25 is deterministic for the same corpus and code. Dense and reranked runs
can differ slightly across hardware and library versions, so replay compares
them within a tolerance.

What is real here: the corpus, queries and relevance judgments are read from
files on disk; the ranking is BM25 over that corpus; the metrics are computed
from the dataset's qrels. The module downloads nothing: it expects a local
directory in the BEIR layout (``corpus.jsonl``, ``queries.jsonl``,
``qrels/<split>.tsv``).

Analyzers (workbench/analysis.py), chosen with ``retrieval_options["analyzer"]``:
``simple`` (the default, a lowercase ``[a-z0-9]+`` split with no stemming or
stopping, so its scores are lower and not comparable with a Lucene index) and
``lucene-english``, which reproduces Lucene's English analysis and BM25 length
encoding. With ``lucene-english`` the top-10 rankings for all 300 judged SciFact
test queries match Pyserini's prebuilt ``beir-v1.0.0-scifact.flat`` index
(k1=0.9, b=0.4); the mean NDCG@10 is 0.6789 and the mean MRR@10 0.6457. The BEIR
paper's 0.665 is a different baseline: Pyserini's multifield index gives 0.6639,
and its flat index, the one this matches, 0.6789.

Only the split's judged queries are evaluated: the queries that appear in
``qrels/<split>.tsv``. BEIR's ``queries.jsonl`` lists every query of every split
(SciFact's also holds its train queries), and BEIR and MTEB score only the
judged ones. They are evaluated in sorted query-ID order, and the first
``spec.query_count`` of them are used (all of them when the count exceeds the
split).
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.ledger import EvidenceLedger, compute_sha256
from .analysis import ANALYZERS, LUCENE_ENGLISH, SIMPLE, analyze, lucene_length_norm, simple_tokens
from .beir_download import file_digests
from .engine import (
    ExecutionResult,
    QueryRecord,
    QuotaExceededViolation,
    enforce_remote_dispatch_grant,
)

REAL_BACKEND = "bm25-stdlib-scifact"
DENSE_BACKEND = "dense-sentence-transformers-scifact"
RERANK_BACKEND = "dense-sentence-transformers+llm-rerank-scifact"
REAL_BACKENDS = (REAL_BACKEND, DENSE_BACKEND, RERANK_BACKEND)
DETERMINISTIC_REAL_BACKENDS = (REAL_BACKEND,)
REAL_NOTICE = (
    "Real retrieval: BM25 (stdlib) over the local BEIR SciFact corpus, scored "
    "against the dataset's qrels. A lexical baseline, not a neural model; its "
    "tokenizer is simpler than Pyserini's, so the BEIR paper's BM25 number is a "
    "sanity neighborhood, not an expected value."
)
DENSE_NOTICE = (
    "Real retrieval: a sentence-transformers embedding model over the local BEIR "
    "corpus, scored against the dataset's qrels. Compare with the model's published "
    "MTEB score before relying on it; results can differ slightly across hardware "
    "and library versions."
)
RERANK_NOTICE = (
    "Real retrieval: a sentence-transformers embedding model over the local BEIR "
    "corpus, with the top candidates reordered by an experimental LLM prompt (0-10 "
    "ratings, not a trained reranker), scored against the dataset's qrels. Results "
    "can differ across hardware and library versions."
)
LUCENE_NOTICE = (
    "Real retrieval: BM25 (stdlib) with a Lucene-style English analyzer over the local BEIR "
    "SciFact corpus, scored against the dataset's qrels. A lexical baseline, not a neural model. "
    "On this corpus its rankings match Pyserini's prebuilt BEIR SciFact flat BM25 index (see the "
    "README); the BEIR paper's BM25 number is a different baseline."
)
NOTICES = {REAL_BACKEND: REAL_NOTICE, DENSE_BACKEND: DENSE_NOTICE, RERANK_BACKEND: RERANK_NOTICE}
BM25_K1 = 0.9
BM25_B = 0.4


def tokenize(text: str) -> List[str]:
    """Lowercase [a-z0-9]+ tokens: no stemming, stopping or field weighting (the default analyzer)."""
    return simple_tokens(text)


def analyzer_of(spec: Any) -> str:
    """The BM25 analyzer a spec selects through ``retrieval_options["analyzer"]`` (default: simple)."""
    return (spec.retrieval_options or {}).get("analyzer", SIMPLE)


def notice_for(spec: Any) -> str:
    """The real-retrieval notice a capsule carries for ``spec``; BM25 runs name their analyzer."""
    label = backend_label(spec)
    if label == REAL_BACKEND and analyzer_of(spec) == LUCENE_ENGLISH:
        return LUCENE_NOTICE
    return NOTICES.get(label, "")


@dataclass
class SciFactData:
    corpus: Dict[str, str]                 # doc_id -> title + " " + text (raw)
    queries: List[Tuple[str, str]]         # judged queries of the split: (query_id, query_text), sorted by ID
    qrels: Dict[str, Dict[str, int]]       # query_id -> doc_id -> grade
    queries_in_file: int = 0               # every query in queries.jsonl, judged or not


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {lineno} is not a JSON object")
            rows.append(row)
    return rows


def load_beir_dataset(data_dir: str, split: str = "test") -> SciFactData:
    """Load a BEIR-format dataset directory (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

    Nothing is downloaded. SciFact ships this layout via the BEIR project
    (github.com/beirproject/beir); the original SciFact release names the
    document field "abstract", which is accepted as a fallback for "text".
    """
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries.jsonl")
    qrels_path = os.path.join(data_dir, "qrels", f"{split}.tsv")
    missing = [p for p in (corpus_path, queries_path, qrels_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "SciFact dataset directory is missing "
            + ", ".join(os.path.relpath(p, data_dir) for p in missing)
            + f" (expected the BEIR layout under {data_dir!r}: corpus.jsonl, queries.jsonl, qrels/{split}.tsv)"
        )

    corpus: Dict[str, str] = {}
    for row in _read_jsonl(corpus_path):
        doc_id = str(row.get("_id", "")).strip()
        if not doc_id:
            raise ValueError(f"{corpus_path} has a row with no '_id'")
        title = str(row.get("title", "") or "")
        text = str(row.get("text", row.get("abstract", "")) or "")
        corpus[doc_id] = f"{title} {text}".strip()

    all_queries: List[Tuple[str, str]] = []
    for row in _read_jsonl(queries_path):
        query_id = str(row.get("_id", "")).strip()
        if not query_id:
            raise ValueError(f"{queries_path} has a row with no '_id'")
        all_queries.append((query_id, str(row.get("text", "") or "")))

    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip("\n").strip()
            if not line:
                continue
            cols = [c.strip() for c in line.split("\t")]
            if lineno == 1 and cols and not cols[0].lstrip("-").isdigit():
                continue  # header row ("query-id")
            if len(cols) == 3:
                query_id, doc_id, score = cols
            elif len(cols) == 4:
                query_id, _iteration, doc_id, score = cols
            else:
                raise ValueError(f"{qrels_path} line {lineno}: expected 3 or 4 tab-separated columns")
            if not score.lstrip("-").isdigit():
                raise ValueError(f"{qrels_path} line {lineno}: score {score!r} is not an integer")
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    # Keep only the queries this split judges; queries.jsonl also lists other splits' queries.
    queries = sorted((q for q in all_queries if q[0] in qrels), key=lambda q: q[0])
    return SciFactData(corpus=corpus, queries=queries, qrels=qrels, queries_in_file=len(all_queries))


class BM25Index:
    """Okapi BM25 over a tokenized corpus, with Lucene-style IDF and deterministic ties."""

    def __init__(self, corpus_tokens: Dict[str, List[str]], k1: float = BM25_K1, b: float = BM25_B,
                 analyzer: str = SIMPLE):
        """``corpus_tokens`` must come from the same ``analyzer`` that ``search`` applies to queries.

        Under ``lucene-english`` the length normalization uses Lucene's one-byte document
        lengths (analysis.lucene_length_norm), while the average length stays exact, as in Lucene.
        """
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer
        self.n_docs = len(corpus_tokens)
        exact_len = {doc_id: len(tokens) for doc_id, tokens in corpus_tokens.items()}
        total = sum(exact_len.values())
        self.avgdl = (total / self.n_docs) if self.n_docs else 0.0
        if analyzer == LUCENE_ENGLISH:
            self.doc_len = {doc_id: lucene_length_norm(n) for doc_id, n in exact_len.items()}
        else:
            self.doc_len = exact_len
        self.postings: Dict[str, Dict[str, int]] = {}
        for doc_id, tokens in corpus_tokens.items():
            counts: Dict[str, int] = {}
            for term in tokens:
                counts[term] = counts.get(term, 0) + 1
            for term, tf in counts.items():
                self.postings.setdefault(term, {})[doc_id] = tf
        self.idf = {
            term: math.log(1 + (self.n_docs - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self.postings.items()
        }

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Rank documents by BM25; ties broken by document ID, so the order is deterministic."""
        scores: Dict[str, float] = {}
        terms = analyze(query_text, self.analyzer)
        if self.analyzer != LUCENE_ENGLISH:
            terms = list(dict.fromkeys(terms))  # the simple analyzer scores each distinct term once
        for term in terms:  # lucene-english scores every occurrence, as Lucene's BooleanQuery does
            post = self.postings.get(term)
            if not post:
                continue
            idf = self.idf[term]
            for doc_id, tf in post.items():
                norm = (self.b * self.doc_len[doc_id] / self.avgdl) if self.avgdl else 0.0
                denom = tf + self.k1 * (1 - self.b + norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[: max(top_k, 0)]


def graded_ndcg_at_10(retrieved_ids: List[str], qrels_for_query: Dict[str, int], k: int = 10) -> float:
    """Graded NDCG@k with 2^grade - 1 gains.

    BEIR's SciFact qrels are binary (every judgment is 1), so on SciFact this
    equals binary NDCG; the grades matter for datasets with graded qrels.
    """
    def gain(grade: int) -> float:
        return float(2 ** grade - 1)

    # Discount 1 / log2(rank + 1): rank 1 counts fully.
    dcg = sum(
        gain(qrels_for_query.get(doc_id, 0)) / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
    )
    ideal = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return (dcg / idcg) if idcg > 0 else 0.0


def mrr_at_10(retrieved_ids: List[str], qrels_for_query: Dict[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first document with a non-zero qrels grade, 0 if none in the top k."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if qrels_for_query.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def backend_label(spec: Any) -> str:
    """Retrieval-backend label for a spec's retrieval_options."""
    options = spec.retrieval_options or {}
    backend = options.get("backend", "bm25")
    if backend == "bm25":
        return REAL_BACKEND
    if backend == "dense":
        return RERANK_BACKEND if options.get("rerank_model") else DENSE_BACKEND
    raise ValueError(f"unknown real retrieval backend {backend!r}")


def build_components(spec: Any, device: Optional[str] = None, batch_size: int = 32):
    """The (retriever, reranker) a dense spec describes; (None, None) for BM25."""
    from .real_models import DenseRetriever, LLMPointwiseReranker

    options = spec.retrieval_options or {}
    if backend_label(spec) == REAL_BACKEND:
        return None, None
    retriever = DenseRetriever(
        model_name=spec.model_id,
        revision=options.get("model_revision"),
        query_prefix=options.get("query_prefix", ""),
        device=device,
        batch_size=batch_size,
    )
    reranker = None
    if options.get("rerank_model"):
        reranker = LLMPointwiseReranker(
            model_id=options["rerank_model"],
            revision=options.get("rerank_revision"),
            top_n=int(options.get("rerank_top_n", 20)),
            device=device,
        )
    return retriever, reranker


def run_real_scifact(
    spec: Any,
    data_dir: str,
    ledger: Optional[EvidenceLedger] = None,
    components: Optional[Tuple[Any, Any]] = None,
    allowed_signers: Optional[str] = None,
) -> ExecutionResult:
    """Runs BM25 over the local SciFact corpus behind the same policy gates as the simulated engine.

    ``spec`` is an ``ExperimentSpec``; ``spec.split`` selects the qrels file and
    ``spec.query_count`` caps how many of the split's queries (in sorted query-ID
    order) are evaluated. The ledger structure mirrors the simulated engine: one
    block per query at timestamps 1000.0 + i, then one consolidation block at
    2000.0. ``components`` overrides the (retriever, reranker) built from the
    spec for dense runs (tests inject stand-ins). ``allowed_signers`` is the
    allowed-signers file that remote-dispatch grants are checked against.
    """
    start_time = time.perf_counter()

    # Invariant Gate 1: query ceiling (same rule as the simulated engine)
    policy = spec.execution_policy
    if spec.query_count > policy.max_query_ceiling:
        raise QuotaExceededViolation(
            f"Requested query count ({spec.query_count}) exceeds policy ceiling ({policy.max_query_ceiling})."
        )

    # Invariant Gate 2: authorization for remote dispatch (same rule as the simulated engine)
    spec_digest = spec.canonical_digest()
    enforce_remote_dispatch_grant(spec, allowed_signers)

    label = backend_label(spec)
    analyzer = analyzer_of(spec)
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}; expected one of {ANALYZERS}")
    if label != REAL_BACKEND and "analyzer" in (spec.retrieval_options or {}):
        raise ValueError("the analyzer option applies to the BM25 backend only")
    data = load_beir_dataset(data_dir, split=spec.split)
    if not data.queries:
        raise ValueError(f"No queries found in {os.path.join(data_dir, 'queries.jsonl')}")

    own_ledger = ledger is None
    ledger = ledger or EvidenceLedger(":memory:")
    try:
        if label == REAL_BACKEND:
            corpus_tokens = {doc_id: analyze(text, analyzer) for doc_id, text in data.corpus.items()}
            index = BM25Index(corpus_tokens, analyzer=analyzer)
            retriever = reranker = None
        else:
            retriever, reranker = components if components is not None else build_components(spec)
            retriever.index(data.corpus)

        selected = data.queries[: spec.query_count] if spec.query_count < len(data.queries) else data.queries

        query_records: List[QueryRecord] = []
        ndcg_list: List[float] = []
        mrr_list: List[float] = []

        for i, (query_id, query_text) in enumerate(selected):
            if label == REAL_BACKEND:
                ranking = index.search(query_text, top_k=spec.top_k)
            else:
                depth = max(spec.top_k, reranker.top_n if reranker else 0)
                ranking = retriever.search(query_text, depth)
                if reranker is not None:
                    ranking = reranker.rerank(query_text, ranking, data.corpus)
                ranking = ranking[: spec.top_k]
            retrieved = [doc_id for doc_id, _score in ranking]
            qrels_for_query = data.qrels.get(query_id, {})

            ndcg = round(graded_ndcg_at_10(retrieved, qrels_for_query), 4)
            mrr = round(mrr_at_10(retrieved, qrels_for_query), 4)
            if qrels_for_query:
                # Queries with no relevant documents are recorded but left out of the means.
                ndcg_list.append(ndcg)
                mrr_list.append(mrr)

            q_digest = compute_sha256(f"{query_id}:{query_text}:{','.join(retrieved)}:{ndcg}:{mrr}")
            query_records.append(
                QueryRecord(
                    query_id=query_id,
                    query_text=query_text,
                    retrieved_doc_ids=retrieved,
                    # qrels grades (0/1/2) per retrieved document, not retriever scores
                    relevance_scores=[float(qrels_for_query.get(doc_id, 0)) for doc_id in retrieved],
                    ndcg_at_10=ndcg,
                    mrr_at_10=mrr,
                    query_digest=q_digest,
                )
            )

            ledger.append(
                domain="EVIDENCE_WORKBENCH",
                action="RETRIEVAL_QUERY_EVAL",
                payload_data={"query_id": query_id, "query_digest": q_digest, "model": spec.model_id},
                metric_data={"ndcg@10": ndcg, "mrr@10": mrr},
                timestamp=1000.0 + float(i),
            )

        mean_ndcg = round(sum(ndcg_list) / len(ndcg_list), 4) if ndcg_list else 0.0
        mean_mrr = round(sum(mrr_list) / len(mrr_list), 4) if mrr_list else 0.0
        ctc = round(1.0 - (mean_ndcg * 0.8 + mean_mrr * 0.2), 4)

        metrics = {
            "NDCG@10": mean_ndcg,
            "MRR@10": mean_mrr,
            "CTC": ctc,
            "queries_evaluated": len(query_records),
        }

        ledger.append(
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

        chain_blocks = ledger.fetch_chain()
        chain_digest = ledger.chain_digest()
    finally:
        if own_ledger:
            ledger.close()

    elapsed = time.perf_counter() - start_time
    env_telemetry: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "pid": os.getpid(),
        "retrieval_backend": label,
        "corpus_documents": len(data.corpus),
        "queries_in_split": len(data.queries),
        "queries_in_file": data.queries_in_file,
        "dataset_dir": os.path.abspath(data_dir),
        "dataset_files": file_digests(data_dir, spec.split),
    }
    if label == REAL_BACKEND:
        env_telemetry.update({"bm25_k1": BM25_K1, "bm25_b": BM25_B})
    else:
        env_telemetry["retriever"] = retriever.environment()
        if reranker is not None:
            env_telemetry["reranker"] = reranker.environment()

    return ExecutionResult(
        spec=spec,
        spec_digest=spec_digest,
        evaluated_queries=query_records,
        metrics=metrics,
        chain_digest=chain_digest,
        ledger_blocks_count=len(chain_blocks),
        environment_telemetry=env_telemetry,
        elapsed_seconds=round(elapsed, 4),
        retrieval_backend=label,
    )


def record_consistency_problems(records: List[Dict[str, Any]], metrics: Dict[str, Any],
                                data: SciFactData) -> List[str]:
    """Check stored per-query records against the qrels and the stored means, without re-running retrieval.

    Replay of a non-deterministic backend accepts rankings that differ from the
    re-run, so it relies on this check to make sure each stored record is
    internally consistent: its NDCG/MRR, grades and digest follow from its own
    ranking and the qrels, and the manifest means follow from the records.
    """
    problems = []
    ndcgs, mrrs = [], []
    for rec in records:
        qid = str(rec.get("query_id"))
        if qid not in data.qrels:
            problems.append(f"record for query {qid} is not a judged query of this split")
            continue
        qrels_for_query = data.qrels[qid]
        retrieved = rec.get("retrieved_doc_ids")
        if not isinstance(retrieved, list) or not all(isinstance(d, str) for d in retrieved):
            problems.append(f"query {qid}: malformed retrieved_doc_ids")
            continue
        ndcg = round(graded_ndcg_at_10(retrieved, qrels_for_query), 4)
        mrr = round(mrr_at_10(retrieved, qrels_for_query), 4)
        if rec.get("ndcg_at_10") != ndcg or rec.get("mrr_at_10") != mrr:
            problems.append(f"query {qid}: stored NDCG/MRR do not match its stored ranking")
        if rec.get("relevance_scores") != [float(qrels_for_query.get(d, 0)) for d in retrieved]:
            problems.append(f"query {qid}: stored relevance grades differ from the qrels")
        expected_digest = compute_sha256(f"{qid}:{rec.get('query_text')}:{','.join(retrieved)}:{ndcg}:{mrr}")
        if rec.get("query_digest") != expected_digest:
            problems.append(f"query {qid}: stored query_digest does not match the record")
        ndcgs.append(ndcg)
        mrrs.append(mrr)
    if ndcgs:
        if round(sum(ndcgs) / len(ndcgs), 4) != metrics.get("NDCG@10"):
            problems.append("manifest NDCG@10 is not the mean of the stored records")
        if round(sum(mrrs) / len(mrrs), 4) != metrics.get("MRR@10"):
            problems.append("manifest MRR@10 is not the mean of the stored records")
    return problems
