"""Zero-dependency signatures for evidence: OpenSSH (SSHSIG) Ed25519 verification.

Signatures use the SSHSIG format from OpenSSH's PROTOCOL.sshsig, so they are made
with an ordinary Ed25519 key through ``ssh-keygen -Y sign`` (or a key in an agent)
and checked here with the standard library alone. Verification is a pure-Python
RFC 8032 check. It runs on public data only, so it needs no constant-time
arithmetic. Signing shells out to ``ssh-keygen``, which keeps the private key out
of this process.

A passing check means the bytes were signed by a key that the allowed-signers
file lists for the given identity and namespace. It does not show who controls
that key. The allowed-signers file and the signing key are the trust root, and
both must be kept outside the party whose records are being checked.

Allowed-signers lines use OpenSSH's format, restricted to what is checked here:
``principal namespaces="ns[,ns]" ssh-ed25519 AAAA...``. Principals match exactly
(no wildcards), ``namespaces=`` is required, and any other option, key type or
wildcard is an error rather than being skipped.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

SIGNATURE_VALID = "VALID"
SIGNATURE_UNSIGNED = "UNSIGNED"
SIGNATURE_INVALID = "INVALID"
SIGNATURE_UNTRUSTED = "UNTRUSTED"

_KEY_TYPE = b"ssh-ed25519"
_SSHSIG_MAGIC = b"SSHSIG"
_SSHSIG_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_SSHSIG_END = "-----END SSH SIGNATURE-----"
_HASH_ALGORITHMS = ("sha256", "sha512")
_KEY_TYPE_PREFIXES = ("ssh-", "ecdsa-", "sk-")

# Curve constants (RFC 8032, section 5.1)
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _recover_x(y: int, sign: int) -> Optional[int]:
    """The x coordinate for y and the parity bit of x, or None if no such point (RFC 8032, 5.1.3)."""
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


# Points are extended coordinates (X, Y, Z, T) with x = X/Z, y = Y/Z and x*y = T/Z.
_BASE_Y = 4 * pow(5, _P - 2, _P) % _P
_BASE_X = _recover_x(_BASE_Y, 0)
_BASE = (_BASE_X, _BASE_Y, 1, _BASE_X * _BASE_Y % _P)


def _point_add(p, q):
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * p[3] * q[3] * _D % _P
    d = 2 * p[2] * q[2] % _P
    e, f, g, h = (b - a) % _P, (d - c) % _P, (d + c) % _P, (b + a) % _P
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _point_mul(s: int, p):
    q = (0, 1, 1, 0)
    while s:
        if s & 1:
            q = _point_add(q, p)
        p = _point_add(p, p)
        s >>= 1
    return q


def _point_encode(p) -> bytes:
    z_inv = pow(p[2], _P - 2, _P)
    x = p[0] * z_inv % _P
    y = p[1] * z_inv % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _point_decode(data: bytes):
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= _P:
        return None
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """RFC 8032 Ed25519 verification: accept when [S]B == R + [H(R, A, M)]A, with S < L."""
    if len(public_key) != 32 or len(signature) != 64:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    a = _point_decode(public_key)
    r = _point_decode(signature[:32])
    if a is None or r is None:
        return False
    k = int.from_bytes(hashlib.sha512(signature[:32] + public_key + message).digest(), "little") % _L
    return _point_encode(_point_mul(s, _BASE)) == _point_encode(_point_add(r, _point_mul(k, a)))


def _string(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def _read_uint32(buf: bytes, offset: int):
    if offset + 4 > len(buf):
        raise ValueError("truncated SSH data")
    return struct.unpack(">I", buf[offset:offset + 4])[0], offset + 4


def _read_string(buf: bytes, offset: int):
    length, offset = _read_uint32(buf, offset)
    if offset + length > len(buf):
        raise ValueError("truncated SSH string")
    return buf[offset:offset + length], offset + length


def _wire_ed25519_key(public_key: bytes) -> bytes:
    return _string(_KEY_TYPE) + _string(public_key)


def _fingerprint(public_key: bytes) -> str:
    digest = hashlib.sha256(_wire_ed25519_key(public_key)).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class SignatureVerdict:
    """The outcome of checking one signature. ``ok`` is true only for VALID."""

    status: str                  # one of the SIGNATURE_* constants
    identity: Optional[str]
    fingerprint: Optional[str]   # SHA256:... of the key that signed, when the signature parsed
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == SIGNATURE_VALID


@dataclass(frozen=True)
class _Signature:
    public_key: bytes
    namespace: str
    hash_algorithm: str
    signature: bytes


@dataclass(frozen=True)
class _AllowedKey:
    principals: frozenset
    namespaces: frozenset
    public_key: bytes


def _parse_sshsig(text: str) -> _Signature:
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0].strip() != _SSHSIG_BEGIN or lines[-1].strip() != _SSHSIG_END:
        raise ValueError("missing SSH SIGNATURE armor")
    try:
        blob = base64.b64decode("".join(line.strip() for line in lines[1:-1]), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("armored body is not base64") from e
    if not blob.startswith(_SSHSIG_MAGIC):
        raise ValueError("bad SSHSIG magic")
    version, offset = _read_uint32(blob, len(_SSHSIG_MAGIC))
    if version != 1:
        raise ValueError(f"unsupported SSHSIG version {version}")
    public_blob, offset = _read_string(blob, offset)
    namespace, offset = _read_string(blob, offset)
    _reserved, offset = _read_string(blob, offset)
    hash_algorithm, offset = _read_string(blob, offset)
    signature_blob, offset = _read_string(blob, offset)
    if offset != len(blob):
        raise ValueError("trailing data after the signature")

    key_type, pos = _read_string(public_blob, 0)
    public_key, pos = _read_string(public_blob, pos)
    if key_type != _KEY_TYPE or len(public_key) != 32 or pos != len(public_blob):
        raise ValueError("signing key is not ssh-ed25519")
    sig_type, pos = _read_string(signature_blob, 0)
    signature, pos = _read_string(signature_blob, pos)
    if sig_type != _KEY_TYPE or len(signature) != 64 or pos != len(signature_blob):
        raise ValueError("signature is not ssh-ed25519")
    if hash_algorithm.decode("ascii", errors="replace") not in _HASH_ALGORITHMS:
        raise ValueError("unsupported hash algorithm")
    try:
        namespace_text = namespace.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("namespace is not UTF-8") from e
    return _Signature(public_key, namespace_text, hash_algorithm.decode("ascii"), signature)


def _signed_data(namespace: str, hash_algorithm: str, message: bytes) -> bytes:
    digest = hashlib.new(hash_algorithm, message).digest()
    return (_SSHSIG_MAGIC + _string(namespace.encode("utf-8")) + _string(b"")
            + _string(hash_algorithm.encode("ascii")) + _string(digest))


def _split_outside_quotes(text: str, separators: str) -> List[str]:
    """Split on separator characters that are not inside double quotes; empty pieces are dropped."""
    pieces: List[str] = []
    current: List[str] = []
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if ch in separators and not quoted:
            if current:
                pieces.append("".join(current))
                current = []
        else:
            current.append(ch)
    if quoted:
        raise ValueError("unterminated quote")
    if current:
        pieces.append("".join(current))
    return pieces


def _unquote(value: str) -> str:
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if '"' in value:
        raise ValueError("malformed quoting in an option value")
    return value


def _load_allowed_signers(path: str) -> List[_AllowedKey]:
    keys: List[_AllowedKey] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                fields = _split_outside_quotes(line, " \t")
            except ValueError as e:
                raise ValueError(f"{path} line {lineno}: {e}") from e
            key_index = next(
                (i for i, field in enumerate(fields) if i >= 1 and field.startswith(_KEY_TYPE_PREFIXES)), None
            )
            if key_index is None or key_index + 1 >= len(fields):
                raise ValueError(f"{path} line {lineno}: no public key found")
            key_type, key_text = fields[key_index], fields[key_index + 1]
            if key_type != _KEY_TYPE.decode("ascii"):
                raise ValueError(f"{path} line {lineno}: unsupported key type {key_type} (only ssh-ed25519)")

            namespaces = None
            try:
                options = _split_outside_quotes(" ".join(fields[1:key_index]), ",")
            except ValueError as e:
                raise ValueError(f"{path} line {lineno}: {e}") from e
            for option in options:
                name, _, value = option.partition("=")
                if name != "namespaces" or namespaces is not None or not value:
                    raise ValueError(f"{path} line {lineno}: unsupported option {name!r}")
                try:
                    unquoted = _unquote(value)
                except ValueError as e:
                    raise ValueError(f"{path} line {lineno}: {e}") from e
                namespaces = frozenset(ns for ns in unquoted.split(",") if ns)
            if not namespaces:
                raise ValueError(f"{path} line {lineno}: namespaces= option is required")

            principals = frozenset(fields[0].split(","))
            if any(not p or any(c in p for c in "*?!") for p in principals):
                raise ValueError(f"{path} line {lineno}: principals must be exact names (no wildcards)")

            try:
                blob = base64.b64decode(key_text, validate=True)
                key_name, pos = _read_string(blob, 0)
                public_key, pos = _read_string(blob, pos)
            except (binascii.Error, ValueError) as e:
                raise ValueError(f"{path} line {lineno}: malformed public key") from e
            if key_name != _KEY_TYPE or len(public_key) != 32 or pos != len(blob):
                raise ValueError(f"{path} line {lineno}: malformed ssh-ed25519 public key")
            keys.append(_AllowedKey(principals, namespaces, public_key))
    return keys


def check_principal(name: str) -> None:
    """Raise ValueError unless ``name`` can be a principal in an allowed-signers file.

    Principals are plain names: no whitespace, commas, quotes or wildcards.
    """
    if not name or any(c.isspace() or c in ',"*?!' for c in name):
        raise ValueError(f"{name!r} must be a plain principal name (no spaces, commas, quotes or wildcards)")


def verify_signature(message: bytes, signature: str, allowed_signers: str, identity: str,
                     namespace: str) -> SignatureVerdict:
    """Check an armored SSHSIG signature over ``message`` (see the module docstring).

    Returns VALID, INVALID (malformed, wrong namespace, or not signed by the
    message's key) or UNTRUSTED (a valid signature by a key that the
    allowed-signers file does not list for this identity and namespace).
    Configuration problems, such as a missing file or an unsupported
    allowed-signers line, raise instead.
    """
    check_principal(identity)
    try:
        parsed = _parse_sshsig(signature)
    except ValueError as e:
        return SignatureVerdict(SIGNATURE_INVALID, identity, None, f"malformed signature: {e}")
    fingerprint = _fingerprint(parsed.public_key)
    if parsed.namespace != namespace:
        return SignatureVerdict(
            SIGNATURE_INVALID, identity, fingerprint,
            f"signed for namespace {parsed.namespace!r}, expected {namespace!r}",
        )
    if not _ed25519_verify(parsed.public_key, _signed_data(namespace, parsed.hash_algorithm, message),
                           parsed.signature):
        return SignatureVerdict(SIGNATURE_INVALID, identity, fingerprint, "signature does not match the content")

    trusted = any(
        identity in key.principals and namespace in key.namespaces and key.public_key == parsed.public_key
        for key in _load_allowed_signers(allowed_signers)
    )
    if not trusted:
        return SignatureVerdict(
            SIGNATURE_UNTRUSTED, identity, fingerprint,
            f"key {fingerprint} is not allowed for {identity!r} in namespace {namespace!r}",
        )
    return SignatureVerdict(SIGNATURE_VALID, identity, fingerprint, f"signed by {identity!r} with {fingerprint}")


def signature_fingerprint(signature: str) -> str:
    """The SHA256 fingerprint of the key that made an armored signature; raises ValueError if malformed."""
    return _fingerprint(_parse_sshsig(signature).public_key)


def sign_bytes(message: bytes, key_path: str, namespace: str) -> str:
    """Sign ``message`` with an OpenSSH key through ``ssh-keygen -Y sign``; returns the armored signature.

    ``key_path`` may be a private key file (ssh-keygen prompts for a passphrase
    if it has one) or a public key file, in which case the key must be loaded in
    ssh-agent.
    """
    keygen = shutil.which("ssh-keygen")
    if keygen is None:
        raise RuntimeError("ssh-keygen was not found; signing needs OpenSSH (ssh-keygen -Y sign)")
    with tempfile.TemporaryDirectory() as tmp:
        message_path = os.path.join(tmp, "message")
        with open(message_path, "wb") as f:
            f.write(message)
        result = subprocess.run(
            [keygen, "-Y", "sign", "-f", key_path, "-n", namespace, message_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ssh-keygen could not sign: {result.stderr.strip()}")
        with open(message_path + ".sig", encoding="utf-8") as f:
            return f.read()
