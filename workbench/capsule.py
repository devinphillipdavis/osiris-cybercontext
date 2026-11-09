"""Evidence capsule packager and parser.

Bundles the experiment spec, the per-query records and a summary into a
.evidence.zip capsule, with a manifest listing each file's SHA-256. The manifest
can be signed with an SSH Ed25519 key (EvidenceCapsule.sign, workbench/signing
rules in core/signing.py). The signature covers manifest.json, and the manifest
lists every other file's checksum, so one passing check covers the whole capsule.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import asdict
from typing import Any, Dict

from core import signing
from core.ledger import compute_sha256
from .engine import ExecutionResult, RETRIEVAL_BACKEND, SIMULATION_NOTICE
from .real_scifact import notice_for
from .schema import canonical_json_bytes, ExperimentSpec


REQUIRED_MANIFEST_KEYS = ("experiment_id", "spec_digest", "chain_digest", "metrics", "spec", "manifest_files")
CHECKSUMMED_FILES = ("raw_queries.json", "summary.md")
# The signature cannot be listed in the manifest it signs, so it is the one extra file a capsule may carry.
SIGNATURE_NAME = "manifest.json.sig"
CAPSULE_NAMESPACE = "dnalang-evidence-capsule"

# summary.md bullet labels -> field names (v1.0.0 labels included)
_SUMMARY_LABELS = {
    "NDCG@10": "NDCG@10",
    "MRR@10": "MRR@10",
    "CTC (derived score)": "CTC",
    "Corpus Task Complexity": "CTC",
    "Queries Evaluated": "queries_evaluated",
    "Canonical Spec Digest": "spec_digest",
    "Ledger Chain Digest": "chain_digest",
    "Merkle Root": "chain_digest",
    "Retrieval Backend": "retrieval_backend",
    "Model": "model_id",
    "Model Label (not loaded)": "model_id",
    "Dataset": "dataset",
    "Dataset Label (placeholder queries)": "dataset",
    "Reranker (experimental)": "rerank_model",
}
_SUMMARY_TITLE = re.compile(r"^# DNA-Lang Evidence Capsule: (?P<value>.*?)\s*$")
_SUMMARY_LINE = re.compile(r"^- \*\*(?P<label>[^*]+?):\*\* `?(?P<value>[^`]*?)`?\s*$")


class EvidenceCapsule:
    """Manages creation, serialization, and inspection of evidence capsules."""

    @staticmethod
    def summary_fields(summary_md: str) -> Dict[str, str]:
        """Parse the figures in a capsule's summary.md into {field: value} strings."""
        fields: Dict[str, str] = {}
        for line in summary_md.splitlines():
            title = _SUMMARY_TITLE.match(line.strip())
            if title and "experiment_id" not in fields:
                fields["experiment_id"] = title.group("value")
            m = _SUMMARY_LINE.match(line.strip())
            if m and m.group("label") in _SUMMARY_LABELS:
                field, value = _SUMMARY_LABELS[m.group("label")], m.group("value").strip()
                if fields.get(field, value) != value:
                    value = "<conflicting duplicate lines>"
                fields[field] = value
        return fields

    @staticmethod
    def pack(result: ExecutionResult, output_path: str) -> str:
        """Packs an ExecutionResult into a self-contained .evidence.zip capsule."""
        # 1. Prepare raw_queries.json
        raw_queries = [asdict(q) for q in result.evaluated_queries]
        raw_queries_bytes = canonical_json_bytes(raw_queries)
        raw_queries_sha = compute_sha256(raw_queries_bytes)

        # 2. Prepare summary.md
        simulated = result.retrieval_backend == RETRIEVAL_BACKEND
        rerank_model = (result.spec.retrieval_options or {}).get("rerank_model")
        notice = SIMULATION_NOTICE if simulated else notice_for(result.spec)
        summary_md = (
            f"# DNA-Lang Evidence Capsule: {result.spec.experiment_id}\n\n"
            + (f"> {notice}\n\n" if notice else "")
            + f"- **Retrieval Backend:** {result.retrieval_backend}\n"
            f"- **Model{' Label (not loaded)' if simulated else ''}:** {result.spec.model_id}\n"
            f"- **Dataset{' Label (placeholder queries)' if simulated else ''}:** {result.spec.dataset} ({result.spec.split})\n"
            + (f"- **Reranker (experimental):** {rerank_model}\n" if rerank_model else "")
            + f"- **Canonical Spec Digest:** `{result.spec_digest}`\n"
            f"- **Ledger Chain Digest:** `{result.chain_digest}`\n"
            f"- **Queries Evaluated:** {result.metrics['queries_evaluated']}\n"
            f"- **NDCG@10:** {result.metrics['NDCG@10']}\n"
            f"- **MRR@10:** {result.metrics['MRR@10']}\n"
            f"- **CTC (derived score):** {result.metrics['CTC']}\n"
            f"- **Execution Time:** {result.elapsed_seconds}s\n"
        )
        summary_md_bytes = summary_md.encode("utf-8")
        summary_md_sha = compute_sha256(summary_md_bytes)

        # 3. Prepare manifest.json
        manifest = {
            "version": "1.1.0",
            "experiment_id": result.spec.experiment_id,
            "retrieval_backend": result.retrieval_backend,
            "notice": notice,
            "spec_digest": result.spec_digest,
            "chain_digest": result.chain_digest,
            "ledger_blocks_count": result.ledger_blocks_count,
            "metrics": result.metrics,
            "environment_telemetry": result.environment_telemetry,
            "elapsed_seconds": result.elapsed_seconds,
            "spec": result.spec.to_dict(),
            "manifest_files": {
                "raw_queries.json": {"sha256": raw_queries_sha, "bytes": len(raw_queries_bytes)},
                "summary.md": {"sha256": summary_md_sha, "bytes": len(summary_md_bytes)},
            },
        }
        manifest_bytes = canonical_json_bytes(manifest)

        # 4. Write zip capsule
        if not output_path.endswith(".evidence.zip") and not output_path.endswith(".zip"):
            output_path += ".evidence.zip"

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", manifest_bytes)
            zf.writestr("raw_queries.json", raw_queries_bytes)
            zf.writestr("summary.md", summary_md_bytes)

        return output_path

    @staticmethod
    def unpack(capsule_path: str) -> Dict[str, Any]:
        """Unpack a capsule and check each file against the checksums in its manifest.

        This catches corruption, edits to a listed file that don't also update
        the manifest, files missing from or added to the capsule, and duplicate
        file names. It does not check manifest.json itself, and an unsigned
        capsule's manifest can be edited freely: someone who edits a file and
        rewrites its checksum produces a capsule that passes. verify_signature
        checks the manifest against a trusted key; ReplayEngine.replay re-runs
        the spec and compares the results.
        """
        if not os.path.exists(capsule_path):
            raise FileNotFoundError(f"Evidence capsule not found: {capsule_path}")

        contents = {}
        with zipfile.ZipFile(capsule_path, "r") as zf:
            names = zf.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Invalid capsule: duplicate file names in the zip")
            for name in names:
                contents[name] = zf.read(name)

        if "manifest.json" not in contents:
            raise ValueError("Invalid capsule: missing manifest.json")

        manifest = json.loads(contents["manifest.json"].decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid capsule: manifest.json is not a JSON object")
        # v1.0.0 capsules stored the chain digest under the old name
        if "chain_digest" not in manifest and "merkle_root" in manifest:
            manifest["chain_digest"] = manifest["merkle_root"]
        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(f"Invalid capsule: manifest is missing {', '.join(missing)}")
        for key in ("metrics", "spec", "manifest_files"):
            if not isinstance(manifest[key], dict):
                raise ValueError(f"Invalid capsule: manifest '{key}' is not a JSON object")

        # Verify internal file integrity
        manifest_files = manifest["manifest_files"]
        for fname in CHECKSUMMED_FILES:
            if fname not in manifest_files:
                raise ValueError(f"Invalid capsule: manifest has no checksum for '{fname}'")
        unlisted = sorted(set(contents) - set(manifest_files) - {"manifest.json", SIGNATURE_NAME})
        if unlisted:
            raise ValueError(f"Invalid capsule: files not listed in the manifest: {', '.join(unlisted)}")
        for fname, meta in manifest_files.items():
            if fname not in contents:
                raise ValueError(f"Corrupted capsule: manifest references missing file '{fname}'")
            actual_sha = compute_sha256(contents[fname])
            if actual_sha != meta["sha256"]:
                raise ValueError(
                    f"Tamper detected in capsule file '{fname}': expected {meta['sha256'][:16]}, found {actual_sha[:16]}"
                )

        raw_queries = json.loads(contents["raw_queries.json"].decode("utf-8"))
        if not isinstance(raw_queries, list) or not all(isinstance(r, dict) for r in raw_queries):
            raise ValueError("Invalid capsule: raw_queries.json is not a list of query records")
        return {
            "manifest": manifest,
            "raw_queries": raw_queries,
            "summary_md": contents["summary.md"].decode("utf-8"),
        }

    @staticmethod
    def sign(capsule_path: str, key_path: str) -> str:
        """Add manifest.json.sig, an SSH Ed25519 signature over manifest.json, to an unsigned capsule.

        The capsule must pass unpack first, so only a consistent capsule is signed.
        The zip is rewritten under a temporary name, so a failed signing leaves the
        capsule unchanged. ``key_path`` is an OpenSSH key, as for ``ssh-keygen -Y sign``.
        Returns the fingerprint of the key that signed.
        """
        EvidenceCapsule.unpack(capsule_path)
        with zipfile.ZipFile(capsule_path, "r") as zf:
            entries = [(info.filename, zf.read(info.filename)) for info in zf.infolist()]
        if SIGNATURE_NAME in dict(entries):
            raise ValueError(f"Capsule is already signed: {SIGNATURE_NAME} is present")
        signature = signing.sign_bytes(dict(entries)["manifest.json"], key_path, CAPSULE_NAMESPACE)

        temp_path = capsule_path + ".signing"
        try:
            with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in entries:
                    zf.writestr(name, data)
                zf.writestr(SIGNATURE_NAME, signature.encode("utf-8"))
            os.replace(temp_path, capsule_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return signing.signature_fingerprint(signature)

    @staticmethod
    def verify_signature(capsule_path: str, allowed_signers: str, identity: str) -> signing.SignatureVerdict:
        """Check manifest.json.sig against an allowed-signers file, after the capsule passes unpack.

        Returns UNSIGNED when the capsule has no signature, and otherwise the
        verdict from core/signing.py. VALID means the manifest is signed by a key
        listed for ``identity``, and every file matches the manifest.
        """
        EvidenceCapsule.unpack(capsule_path)
        with zipfile.ZipFile(capsule_path, "r") as zf:
            names = set(zf.namelist())
            if SIGNATURE_NAME not in names:
                return signing.SignatureVerdict(
                    signing.SIGNATURE_UNSIGNED, None, None, f"no {SIGNATURE_NAME} in the capsule"
                )
            manifest_bytes = zf.read("manifest.json")
            signature = zf.read(SIGNATURE_NAME).decode("utf-8", errors="replace")
        return signing.verify_signature(manifest_bytes, signature, allowed_signers, identity, CAPSULE_NAMESPACE)
