"""Unit tests for the zero-dependency SSH signature verifier (core/signing.py)."""

import os
import shutil
import subprocess
import tempfile
import unittest

from core import signing

NAMESPACE = "dnalang-test-namespace"
HAVE_SSH_KEYGEN = shutil.which("ssh-keygen") is not None

# RFC 8032, section 7.1 test vectors: (public key, message, signature), hex encoded.
RFC8032_VECTORS = [
    (
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]


class TestEd25519Core(unittest.TestCase):
    def test_base_point_encoding(self):
        self.assertEqual(signing._point_encode(signing._BASE).hex(), "58" + "66" * 31)

    def test_rfc8032_vectors_verify(self):
        for public_hex, message_hex, signature_hex in RFC8032_VECTORS:
            with self.subTest(message=message_hex):
                self.assertTrue(signing._ed25519_verify(
                    bytes.fromhex(public_hex), bytes.fromhex(message_hex), bytes.fromhex(signature_hex)))

    def test_rejects_altered_message_signature_and_key(self):
        public_hex, message_hex, signature_hex = RFC8032_VECTORS[1]
        public, message, signature = (bytes.fromhex(x) for x in (public_hex, message_hex, signature_hex))
        self.assertFalse(signing._ed25519_verify(public, b"\x73", signature))
        self.assertFalse(signing._ed25519_verify(public, message, bytes([signature[0] ^ 1]) + signature[1:]))
        other_key = bytes.fromhex(RFC8032_VECTORS[0][0])
        self.assertFalse(signing._ed25519_verify(other_key, message, signature))

    def test_rejects_non_canonical_scalar(self):
        public_hex, message_hex, signature_hex = RFC8032_VECTORS[1]
        public, message, signature = (bytes.fromhex(x) for x in (public_hex, message_hex, signature_hex))
        s = int.from_bytes(signature[32:], "little") + signing._L
        forged = signature[:32] + s.to_bytes(32, "little")
        self.assertFalse(signing._ed25519_verify(public, message, forged))


@unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen (OpenSSH) is not installed")
class TestSshSignatures(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.key = self._make_key("key")
        self.other_key = self._make_key("other")
        self.allowed = os.path.join(self.dir, "allowed_signers")
        self._write_allowed(f'alice namespaces="{NAMESPACE}" {self._pub(self.key)}')

    def _make_key(self, name):
        path = os.path.join(self.dir, name)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", path], check=True)
        return path

    @staticmethod
    def _pub(key_path):
        with open(key_path + ".pub", encoding="utf-8") as f:
            return f.read().strip()

    def _write_allowed(self, text):
        with open(self.allowed, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    def _sign(self, message, key=None, namespace=NAMESPACE, hashalg=None):
        key = key or self.key
        path = os.path.join(self.dir, "message")
        with open(path, "wb") as f:
            f.write(message)
        if os.path.exists(path + ".sig"):
            os.remove(path + ".sig")
        command = ["ssh-keygen", "-Y", "sign", "-f", key, "-n", namespace]
        if hashalg:
            command += ["-O", f"hashalg={hashalg}"]
        subprocess.run(command + [path], check=True, capture_output=True)
        with open(path + ".sig", encoding="utf-8") as f:
            return f.read()

    def test_openssh_signatures_verify_across_sizes_and_hashes(self):
        for size in (0, 1, 55, 56, 63, 64, 111, 112, 128, 1000, 5000):
            message = bytes((i * 37 + size) % 256 for i in range(size))
            for hashalg in ("sha512", "sha256"):
                with self.subTest(size=size, hashalg=hashalg):
                    sig = self._sign(message, hashalg=hashalg)
                    verdict = signing.verify_signature(message, sig, self.allowed, "alice", NAMESPACE)
                    self.assertEqual(verdict.status, signing.SIGNATURE_VALID, verdict.detail)
                    self.assertTrue(verdict.fingerprint.startswith("SHA256:"))

    def test_signing_helper_round_trips(self):
        sig = signing.sign_bytes(b"manifest bytes", self.key, NAMESPACE)
        verdict = signing.verify_signature(b"manifest bytes", sig, self.allowed, "alice", NAMESPACE)
        self.assertTrue(verdict.ok, verdict.detail)
        self.assertEqual(verdict.identity, "alice")

    def test_altered_content_is_invalid(self):
        sig = self._sign(b"original")
        verdict = signing.verify_signature(b"original!", sig, self.allowed, "alice", NAMESPACE)
        self.assertEqual(verdict.status, signing.SIGNATURE_INVALID)

    def test_wrong_namespace_is_invalid(self):
        sig = self._sign(b"content", namespace="some-other-namespace")
        verdict = signing.verify_signature(b"content", sig, self.allowed, "alice", NAMESPACE)
        self.assertEqual(verdict.status, signing.SIGNATURE_INVALID)
        self.assertIn("namespace", verdict.detail)

    def test_unlisted_key_is_untrusted(self):
        sig = self._sign(b"content", key=self.other_key)
        verdict = signing.verify_signature(b"content", sig, self.allowed, "alice", NAMESPACE)
        self.assertEqual(verdict.status, signing.SIGNATURE_UNTRUSTED)

    def test_unknown_identity_is_untrusted(self):
        sig = self._sign(b"content")
        verdict = signing.verify_signature(b"content", sig, self.allowed, "mallory", NAMESPACE)
        self.assertEqual(verdict.status, signing.SIGNATURE_UNTRUSTED)

    def test_namespaces_option_limits_the_key(self):
        self._write_allowed(f'alice namespaces="only-this-one" {self._pub(self.key)}')
        sig = self._sign(b"content")
        verdict = signing.verify_signature(b"content", sig, self.allowed, "alice", NAMESPACE)
        self.assertEqual(verdict.status, signing.SIGNATURE_UNTRUSTED)

    def test_quoted_namespace_list_admits_each_namespace(self):
        self._write_allowed(f'alice namespaces="other-one,{NAMESPACE}" {self._pub(self.key)}')
        sig = self._sign(b"content")
        verdict = signing.verify_signature(b"content", sig, self.allowed, "alice", NAMESPACE)
        self.assertEqual(verdict.status, signing.SIGNATURE_VALID, verdict.detail)

    def test_namespace_list_must_be_quoted(self):
        self._write_allowed(f"alice namespaces=one,two {self._pub(self.key)}")
        sig = self._sign(b"content")
        with self.assertRaises(ValueError):
            signing.verify_signature(b"content", sig, self.allowed, "alice", NAMESPACE)

    def test_malformed_signature_is_invalid(self):
        for text in ("", "not a signature", "-----BEGIN SSH SIGNATURE-----\n!!!\n-----END SSH SIGNATURE-----"):
            with self.subTest(text=text[:20]):
                verdict = signing.verify_signature(b"content", text, self.allowed, "alice", NAMESPACE)
                self.assertEqual(verdict.status, signing.SIGNATURE_INVALID)

    def test_allowed_signers_config_errors_raise(self):
        pub = self._pub(self.key)
        bad_lines = {
            "unsupported option": f'alice namespaces="{NAMESPACE}",cert-authority {pub}',
            "wildcard principal": f'*@example namespaces="{NAMESPACE}" {pub}',
            "missing namespaces": f"alice {pub}",
            "unsupported key type": "alice namespaces=\"x\" ssh-rsa AAAAB3NzaC1yc2E=",
        }
        sig = self._sign(b"content")
        for label, line in bad_lines.items():
            with self.subTest(label=label):
                self._write_allowed(line)
                with self.assertRaises(ValueError):
                    signing.verify_signature(b"content", sig, self.allowed, "alice", NAMESPACE)

    def test_missing_allowed_signers_file_raises(self):
        sig = self._sign(b"content")
        with self.assertRaises(FileNotFoundError):
            signing.verify_signature(b"content", sig, os.path.join(self.dir, "absent"), "alice", NAMESPACE)

    def test_empty_identity_is_rejected(self):
        sig = self._sign(b"content")
        with self.assertRaises(ValueError):
            signing.verify_signature(b"content", sig, self.allowed, "", NAMESPACE)


if __name__ == "__main__":
    unittest.main()
