"""
sync_server.py — Lightweight Flask relay for FaceVault federated sync.

This is a *blind relay* in the semi-honest model:
  - It receives encrypted sync manifests from multiple devices.
  - For comparison it decrypts centroids using the device-provided key.
  - It computes pairwise cosine similarity and flags potential matches.
  - It never stores or forwards raw images.

Run standalone:
    python sync_server.py

Production note:
    In a production deployment the comparison would happen via homomorphic
    encryption or secure multi-party computation so the relay never sees
    plaintext centroids.  See privacy.py docstring for details.
"""

import sys
import json
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, request, jsonify

from privacy import decrypt_manifest, manifest_from_json, SyncManifest
from config import SYNC_SERVER_HOST, SYNC_SERVER_PORT, SYNC_MATCH_THRESHOLD

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# In-memory store  (reset every server restart — fine for a demo)
# ──────────────────────────────────────────────────────────────────────────────

manifests: Dict[str, SyncManifest] = {}        # device_id → latest manifest
merge_results: Dict[str, List[dict]] = {}      # device_id → list of suggestions


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/submit_manifest", methods=["POST"])
def submit_manifest():
    """Receive an encrypted sync manifest from a device."""
    try:
        raw = request.get_data(as_text=True)
        manifest = manifest_from_json(raw)
        manifests[manifest.device_id] = manifest
        return jsonify({
            "status": "ok",
            "device_id": manifest.device_id,
            "clusters_received": len(manifest.cluster_ids),
            "devices_registered": len(manifests),
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/compare", methods=["GET"])
def compare():
    """Compare centroids across all registered devices.

    Decrypts each manifest, computes pairwise cosine similarity between
    centroids from *different* devices, and returns match pairs that
    exceed SYNC_MATCH_THRESHOLD.
    """
    if len(manifests) < 2:
        return jsonify({
            "status": "waiting",
            "message": f"Need ≥ 2 devices, currently have {len(manifests)}.",
        }), 200

    # Decrypt all manifests
    device_centroids: Dict[str, Dict[int, np.ndarray]] = {}
    for dev_id, m in manifests.items():
        key = m.key.encode("utf-8")
        device_centroids[dev_id] = decrypt_manifest(m.ciphertext, key)

    # Pairwise cross-device comparison
    dev_ids = sorted(device_centroids.keys())
    matches: List[dict] = []

    for i, dev_a in enumerate(dev_ids):
        for dev_b in dev_ids[i + 1:]:
            cents_a = device_centroids[dev_a]
            cents_b = device_centroids[dev_b]
            for cid_a, vec_a in cents_a.items():
                for cid_b, vec_b in cents_b.items():
                    sim = float(np.dot(vec_a, vec_b))
                    if sim >= SYNC_MATCH_THRESHOLD:
                        match = {
                            "device_a": dev_a,
                            "cluster_a": cid_a,
                            "device_b": dev_b,
                            "cluster_b": cid_b,
                            "similarity": round(sim, 4),
                        }
                        matches.append(match)
                        # Store per-device suggestions
                        merge_results.setdefault(dev_a, []).append(match)
                        merge_results.setdefault(dev_b, []).append(match)

    return jsonify({
        "status": "ok",
        "total_matches": len(matches),
        "matches": matches,
    }), 200


@app.route("/merge_suggestions/<device_id>", methods=["GET"])
def merge_suggestions(device_id: str):
    """Return merge suggestions for a specific device."""
    suggestions = merge_results.get(device_id, [])
    return jsonify({
        "status": "ok",
        "device_id": device_id,
        "suggestions": suggestions,
    }), 200


@app.route("/status", methods=["GET"])
def status():
    """Health check + summary."""
    return jsonify({
        "status": "ok",
        "devices_registered": list(manifests.keys()),
        "total_suggestions": sum(len(v) for v in merge_results.values()),
    }), 200


@app.route("/reset", methods=["POST"])
def reset():
    """Clear all stored manifests and results."""
    manifests.clear()
    merge_results.clear()
    return jsonify({"status": "ok", "message": "Server state reset."}), 200


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🔐 FaceVault Sync Server starting on {SYNC_SERVER_HOST}:{SYNC_SERVER_PORT}")
    print("   Semi-honest relay — see privacy.py for production notes.")
    print("   Press Ctrl+C to stop.\n")
    app.run(host=SYNC_SERVER_HOST, port=SYNC_SERVER_PORT, debug=True)
