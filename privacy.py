"""
privacy.py — Differential-privacy and encryption layer for FaceVault sync.

Applies calibrated Laplacian noise to centroids and wraps them in
Fernet (AES-128-CBC) symmetric encryption before export.

NOTE — Production path
------
In a production system the relay should **never** see plaintext centroids.
This would require homomorphic encryption (e.g. CKKS via TenSEAL) or secure
multi-party computation.  For this project we use a *semi-honest* relay
model: the server decrypts centroids for comparison but is trusted not to
leak them.  This is documented explicitly in the README and submission.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import msgpack
from cryptography.fernet import Fernet

from config import DP_EPSILON, EMBEDDING_DIM


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SyncManifest:
    """Encrypted centroid package ready for transmission."""
    device_id: str
    timestamp: float
    cluster_ids: list          # list of int — which clusters are included
    ciphertext: bytes          # Fernet-encrypted msgpack payload
    key: str                   # Fernet key (base64 str) — in prod this is pre-shared / DH


# ──────────────────────────────────────────────────────────────────────────────
# Differential Privacy
# ──────────────────────────────────────────────────────────────────────────────

def add_dp_noise(
    centroid: np.ndarray,
    epsilon: float = DP_EPSILON,
    cluster_size: int = 1,
) -> np.ndarray:
    """Add calibrated Laplacian noise to a centroid vector.

    Parameters
    ----------
    centroid     : 1-D float32 array (512-dim)
    epsilon      : privacy budget — lower ε → more noise → more private
    cluster_size : number of images in the cluster (affects sensitivity)

    Returns
    -------
    Noisy centroid (same shape, re-normalised to unit length).

    Sensitivity = 2 / n   (L2-normalised vectors have max pairwise dist = 2)
    Scale (b)  = sensitivity / ε
    """
    sensitivity = 2.0 / max(cluster_size, 1)
    scale = sensitivity / max(epsilon, 1e-6)
    noise = np.random.laplace(loc=0.0, scale=scale, size=centroid.shape).astype("float32")
    noisy = centroid + noise
    # Re-normalise to stay on the unit hypersphere
    norm = np.linalg.norm(noisy)
    if norm > 0:
        noisy = noisy / norm
    return noisy


# ──────────────────────────────────────────────────────────────────────────────
# Encryption helpers
# ──────────────────────────────────────────────────────────────────────────────

def _centroids_to_bytes(centroids: Dict[int, np.ndarray]) -> bytes:
    """Serialise centroid dict → msgpack bytes."""
    payload = {str(k): v.tolist() for k, v in centroids.items()}
    return msgpack.packb(payload, use_bin_type=True)


def _bytes_to_centroids(raw: bytes) -> Dict[int, np.ndarray]:
    """Deserialise msgpack bytes → centroid dict."""
    payload = msgpack.unpackb(raw, raw=False)
    return {int(k): np.array(v, dtype="float32") for k, v in payload.items()}


def encrypt_manifest(centroids: Dict[int, np.ndarray], key: bytes) -> bytes:
    """Encrypt serialised centroids with Fernet."""
    fernet = Fernet(key)
    raw = _centroids_to_bytes(centroids)
    return fernet.encrypt(raw)


def decrypt_manifest(ciphertext: bytes, key: bytes) -> Dict[int, np.ndarray]:
    """Decrypt Fernet ciphertext back to centroid dict."""
    fernet = Fernet(key)
    raw = fernet.decrypt(ciphertext)
    return _bytes_to_centroids(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_sync_manifest(
    device_id: str,
    centroids: Dict[int, np.ndarray],
    cluster_sizes: Optional[Dict[int, int]] = None,
    epsilon: float = DP_EPSILON,
    key: Optional[bytes] = None,
) -> SyncManifest:
    """Full pipeline: apply DP noise → encrypt → return SyncManifest.

    Parameters
    ----------
    device_id     : unique string identifying this device
    centroids     : {cluster_id: 512-d np.ndarray}
    cluster_sizes : {cluster_id: int} — number of images per cluster
    epsilon       : DP budget
    key           : Fernet key; one is generated if not provided

    Returns
    -------
    SyncManifest dataclass
    """
    if key is None:
        key = Fernet.generate_key()

    if cluster_sizes is None:
        cluster_sizes = {k: 1 for k in centroids}

    # Apply DP noise to each centroid
    noisy: Dict[int, np.ndarray] = {}
    for c_id, vec in centroids.items():
        noisy[c_id] = add_dp_noise(vec, epsilon=epsilon, cluster_size=cluster_sizes.get(c_id, 1))

    # Encrypt
    ciphertext = encrypt_manifest(noisy, key)

    return SyncManifest(
        device_id=device_id,
        timestamp=time.time(),
        cluster_ids=sorted(noisy.keys()),
        ciphertext=ciphertext,
        key=key.decode("utf-8"),
    )


def manifest_to_json(manifest: SyncManifest) -> str:
    """Serialise a SyncManifest to a JSON string (for HTTP transport)."""
    d = asdict(manifest)
    # ciphertext is bytes — encode as latin-1 for JSON safety
    d["ciphertext"] = d["ciphertext"].decode("latin-1")
    return json.dumps(d)


def manifest_from_json(raw: str) -> SyncManifest:
    """Deserialise a JSON string back to a SyncManifest."""
    d = json.loads(raw)
    d["ciphertext"] = d["ciphertext"].encode("latin-1")
    return SyncManifest(**d)
