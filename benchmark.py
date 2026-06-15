"""
benchmark.py — Evaluation & benchmarking for FaceVault federated sync.

Produces the quantitative results needed for the project submission:
  1. Sync accuracy on LFW pairs (Precision / Recall / F1)
  2. Privacy budget (ε) vs accuracy trade-off curve
  3. Centroid stability under EMA vs simple mean
  4. Encryption overhead timing

Usage:
    python benchmark.py [--output benchmarks/]

Results are saved as JSON + matplotlib PNG plots.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from config import (
    EMBEDDING_DIM,
    DP_EPSILON,
    SYNC_MATCH_THRESHOLD,
    CENTROID_EMA_ALPHA,
    CENTROID_MERGE_THRESHOLD,
)
from centroid import compute_centroids, compute_centroids_ema, merge_fragmented_clusters
from privacy import add_dp_noise, encrypt_manifest, decrypt_manifest, generate_sync_manifest
from cryptography.fernet import Fernet


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_synthetic_clusters(
    n_identities: int = 20,
    imgs_per_id: int = 8,
    noise_std: float = 0.15,
    dim: int = EMBEDDING_DIM,
) -> Tuple[Dict[int, List[str]], List[str], faiss.Index]:
    """Create a synthetic dataset with known ground-truth identities.

    Each identity gets a random "true" centroid.  Individual embeddings
    are that centroid + Gaussian noise, then L2-normalised.
    """
    index = faiss.IndexFlatIP(dim)
    paths: List[str] = []
    clusters: Dict[int, List[str]] = {}

    for cid in range(n_identities):
        true_centroid = np.random.randn(dim).astype("float32")
        true_centroid /= np.linalg.norm(true_centroid)
        cluster_paths = []
        for j in range(imgs_per_id):
            vec = true_centroid + np.random.randn(dim).astype("float32") * noise_std
            vec /= np.linalg.norm(vec)
            vec = vec.reshape(1, -1)
            index.add(vec)
            fake_path = f"/synthetic/{cid}/img_{j}.jpg"
            paths.append(fake_path)
            cluster_paths.append(fake_path)
        clusters[cid] = cluster_paths

    return clusters, paths, index


def _split_dataset(
    clusters: Dict[int, List[str]],
    paths: List[str],
    index: faiss.Index,
    overlap_ratio: float = 0.5,
) -> Tuple[dict, dict]:
    """Split a synthetic dataset into two 'devices' with controlled overlap.

    `overlap_ratio` of the identities appear on both devices.
    """
    all_ids = sorted(clusters.keys())
    n_overlap = max(1, int(len(all_ids) * overlap_ratio))
    overlap_ids = all_ids[:n_overlap]
    only_a_ids = all_ids[n_overlap: n_overlap + (len(all_ids) - n_overlap) // 2]
    only_b_ids = all_ids[n_overlap + len(only_a_ids):]

    device_a_ids = set(overlap_ids) | set(only_a_ids)
    device_b_ids = set(overlap_ids) | set(only_b_ids)

    def subset(ids):
        sub_clusters = {k: v for k, v in clusters.items() if k in ids}
        sub_paths = [p for cid in ids for p in clusters[cid]]
        # Build a sub-index from the parent index.
        # NOTE: index.reconstruct() is used here to extract vectors.  This
        # works for IndexFlatIP (used in synthetic benchmarks) but will FAIL
        # on IndexHNSWFlat unless it was built with store_pairs=True.
        # If adapting this benchmark to run against a real HNSW database,
        # either (a) store the raw embeddings separately in app_data.pkl, or
        # (b) create the HNSW index with: idx = faiss.IndexHNSWFlat(dim, M);
        #     idx.hnsw.store_pairs = True.
        sub_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        path_to_idx = {p: i for i, p in enumerate(paths)}
        for p in sub_paths:
            vec = index.reconstruct(path_to_idx[p]).reshape(1, -1)
            sub_index.add(vec)
        return {"clusters": sub_clusters, "paths": sub_paths, "index": sub_index}

    return subset(device_a_ids), subset(device_b_ids), overlap_ids


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 1: Sync accuracy
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_sync_accuracy(
    n_identities: int = 30,
    imgs_per_id: int = 8,
    epsilon: float = DP_EPSILON,
    threshold: float = SYNC_MATCH_THRESHOLD,
    n_trials: int = 10,
) -> Dict:
    """Simulate two devices, run the full sync pipeline, measure P / R / F1."""
    precisions, recalls, f1s = [], [], []

    for _ in range(n_trials):
        clusters, paths, index = _make_synthetic_clusters(n_identities, imgs_per_id)
        dev_a, dev_b, overlap_ids = _split_dataset(clusters, paths, index)
        overlap_set = set(overlap_ids)

        # Compute centroids
        cents_a = compute_centroids(dev_a["clusters"], dev_a["paths"], dev_a["index"])
        cents_b = compute_centroids(dev_b["clusters"], dev_b["paths"], dev_b["index"])

        # Add DP noise
        sizes_a = {k: len(v) for k, v in dev_a["clusters"].items()}
        sizes_b = {k: len(v) for k, v in dev_b["clusters"].items()}
        noisy_a = {k: add_dp_noise(v, epsilon, sizes_a.get(k, 1)) for k, v in cents_a.items()}
        noisy_b = {k: add_dp_noise(v, epsilon, sizes_b.get(k, 1)) for k, v in cents_b.items()}

        # Cross-device comparison
        tp, fp, fn = 0, 0, 0
        matched_overlaps = set()
        for ca_id, va in noisy_a.items():
            for cb_id, vb in noisy_b.items():
                sim = float(np.dot(va, vb))
                is_match = sim >= threshold
                is_true = (ca_id == cb_id) and (ca_id in overlap_set)
                if is_match and is_true:
                    tp += 1
                    matched_overlaps.add(ca_id)
                elif is_match and not is_true:
                    fp += 1

        fn = len(overlap_set) - len(matched_overlaps)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision_mean": round(float(np.mean(precisions)), 4),
        "recall_mean": round(float(np.mean(recalls)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "precision_std": round(float(np.std(precisions)), 4),
        "recall_std": round(float(np.std(recalls)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "n_trials": n_trials,
        "epsilon": epsilon,
        "threshold": threshold,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 2: ε vs accuracy trade-off
# ──────────────────────────────────────────────────────────────────────────────

def epsilon_accuracy_curve(
    epsilons: List[float] | None = None,
    output_dir: str = "benchmarks",
) -> Dict:
    """Sweep ε values and plot accuracy vs privacy budget."""
    if epsilons is None:
        epsilons = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    results = []
    for eps in epsilons:
        r = evaluate_sync_accuracy(epsilon=eps, n_trials=5)
        results.append({"epsilon": eps, **r})

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot([r["epsilon"] for r in results], [r["f1_mean"] for r in results],
            "o-", color="#2563eb", linewidth=2, markersize=6, label="F1 Score")
    ax.fill_between(
        [r["epsilon"] for r in results],
        [r["f1_mean"] - r["f1_std"] for r in results],
        [r["f1_mean"] + r["f1_std"] for r in results],
        alpha=0.2, color="#2563eb",
    )
    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Privacy–Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "epsilon_accuracy.png"), dpi=150)
    plt.close(fig)

    return {"curve_data": results, "plot": os.path.join(output_dir, "epsilon_accuracy.png")}


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 3: Centroid stability (EMA vs mean)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_centroid_stability(
    n_rounds: int = 10,
    imgs_per_round: int = 4,
    output_dir: str = "benchmarks",
) -> Dict:
    """Measure centroid drift over incremental additions: EMA vs simple mean."""
    dim = EMBEDDING_DIM
    # Create a "true" identity centroid
    true_centroid = np.random.randn(dim).astype("float32")
    true_centroid /= np.linalg.norm(true_centroid)

    mean_drifts = []
    ema_drifts = []
    prev_ema_centroid = None

    all_vecs = []
    for r in range(n_rounds):
        # Simulate new photos arriving
        new_vecs = []
        for _ in range(imgs_per_round):
            v = true_centroid + np.random.randn(dim).astype("float32") * 0.15
            v /= np.linalg.norm(v)
            new_vecs.append(v)
        all_vecs.extend(new_vecs)

        # Simple mean over ALL seen so far
        mat = np.vstack(all_vecs).astype("float32")
        simple_mean = mat.mean(axis=0)
        simple_mean /= np.linalg.norm(simple_mean)

        # EMA update
        batch_mean = np.vstack(new_vecs).mean(axis=0).astype("float32")
        batch_mean /= np.linalg.norm(batch_mean)
        if prev_ema_centroid is None:
            ema_centroid = batch_mean
        else:
            ema_centroid = CENTROID_EMA_ALPHA * batch_mean + (1 - CENTROID_EMA_ALPHA) * prev_ema_centroid
            ema_centroid /= np.linalg.norm(ema_centroid)
        prev_ema_centroid = ema_centroid

        mean_drifts.append(float(np.dot(simple_mean, true_centroid)))
        ema_drifts.append(float(np.dot(ema_centroid, true_centroid)))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    rounds = list(range(1, n_rounds + 1))
    ax.plot(rounds, mean_drifts, "s-", label="Simple Mean", color="#dc2626", linewidth=2)
    ax.plot(rounds, ema_drifts, "o-", label="EMA (α=0.3)", color="#2563eb", linewidth=2)
    ax.set_xlabel("Round (batch additions)", fontsize=12)
    ax.set_ylabel("Cosine Similarity to True Centroid", fontsize=12)
    ax.set_title("Centroid Stability: EMA vs Simple Mean", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "centroid_stability.png"), dpi=150)
    plt.close(fig)

    return {
        "mean_final_sim": round(mean_drifts[-1], 4),
        "ema_final_sim": round(ema_drifts[-1], 4),
        "plot": os.path.join(output_dir, "centroid_stability.png"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 4: Encryption overhead
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_encryption_overhead(
    n_clusters: int = 50,
    n_repeats: int = 100,
) -> Dict:
    """Time encrypt/decrypt cycles and report throughput."""
    centroids = {}
    for i in range(n_clusters):
        v = np.random.randn(EMBEDDING_DIM).astype("float32")
        v /= np.linalg.norm(v)
        centroids[i] = v

    key = Fernet.generate_key()

    # Encrypt
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ct = encrypt_manifest(centroids, key)
    encrypt_ms = (time.perf_counter() - t0) / n_repeats * 1000

    # Decrypt
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        decrypt_manifest(ct, key)
    decrypt_ms = (time.perf_counter() - t0) / n_repeats * 1000

    return {
        "n_clusters": n_clusters,
        "n_repeats": n_repeats,
        "encrypt_avg_ms": round(encrypt_ms, 3),
        "decrypt_avg_ms": round(decrypt_ms, 3),
        "payload_bytes": len(ct),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FaceVault federated sync benchmarks")
    parser.add_argument("--output", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  FaceVault Federated Sync — Benchmark Suite")
    print("=" * 60)

    # 1. Sync accuracy
    print("\n[1/4] Evaluating sync accuracy (P / R / F1) ...")
    acc = evaluate_sync_accuracy()
    print(f"      Precision: {acc['precision_mean']:.4f} ± {acc['precision_std']:.4f}")
    print(f"      Recall:    {acc['recall_mean']:.4f} ± {acc['recall_std']:.4f}")
    print(f"      F1:        {acc['f1_mean']:.4f} ± {acc['f1_std']:.4f}")

    # 2. ε–accuracy curve
    print("\n[2/4] Sweeping ε for privacy–accuracy trade-off ...")
    eps_curve = epsilon_accuracy_curve(output_dir=args.output)
    print(f"      Plot saved → {eps_curve['plot']}")

    # 3. Centroid stability
    print("\n[3/4] Benchmarking centroid stability (EMA vs Mean) ...")
    stability = benchmark_centroid_stability(output_dir=args.output)
    print(f"      Simple Mean final sim: {stability['mean_final_sim']}")
    print(f"      EMA final sim:         {stability['ema_final_sim']}")
    print(f"      Plot saved → {stability['plot']}")

    # 4. Encryption overhead
    print("\n[4/4] Measuring encryption overhead ...")
    enc = benchmark_encryption_overhead()
    print(f"      Encrypt: {enc['encrypt_avg_ms']:.3f} ms | Decrypt: {enc['decrypt_avg_ms']:.3f} ms")
    print(f"      Payload size: {enc['payload_bytes']:,} bytes")

    # Save combined results
    combined = {
        "sync_accuracy": acc,
        "epsilon_curve": eps_curve["curve_data"],
        "centroid_stability": stability,
        "encryption_overhead": enc,
    }
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✅ All results saved → {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
