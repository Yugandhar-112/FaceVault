"""
benchmark.py — Evaluation & benchmarking for FaceVault federated sync.

Produces the quantitative results needed for the project submission:
  1. Sync accuracy on LFW pairs (Precision / Recall / F1)
  2. Privacy budget (ε) vs accuracy trade-off curve
  3. Centroid stability under EMA vs simple mean
  4. Encryption overhead timing
  5. HNSW Search Latency Benchmark
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
    """Create a synthetic dataset with known ground-truth identities."""
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
    """Split dataset into two devices."""
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
        sub_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        path_to_idx = {p: i for i, p in enumerate(paths)}
        for p in sub_paths:
            vec = index.reconstruct(path_to_idx[p]).reshape(1, -1)
            sub_index.add(vec)
        return {"clusters": sub_clusters, "paths": sub_paths, "index": sub_index}

    return subset(device_a_ids), subset(device_b_ids), overlap_ids


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 1–4: Existing Benchmarks
# ──────────────────────────────────────────────────────────────────────────────
# [Functions evaluate_sync_accuracy, epsilon_accuracy_curve, 
#  benchmark_centroid_stability, benchmark_encryption_overhead remain same]

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 5: HNSW Latency (NEW)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_hnsw_latency(n_vectors: int = 13233, n_queries: int = 1000) -> Dict:
    """Benchmark HNSW search latency."""
    dim = EMBEDDING_DIM
    # Build HNSW index
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    data = np.random.rand(n_vectors, dim).astype("float32")
    faiss.normalize_L2(data)
    index.add(data)
    
    # Queries
    queries = np.random.rand(n_queries, dim).astype("float32")
    faiss.normalize_L2(queries)
    index.hnsw.efSearch = 32

    # Warm-up
    _ = index.search(queries[0:1], k=5)

    # Benchmark
    start = time.perf_counter()
    index.search(queries, k=5)
    end = time.perf_counter()

    avg_ms = ((end - start) * 1000) / n_queries
    return {"n_vectors": n_vectors, "avg_query_ms": round(avg_ms, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FaceVault federated sync benchmarks")
    parser.add_argument("--output", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("   FaceVault Federated Sync — Benchmark Suite")
    print("=" * 60)

    # Existing Benchmarks 1-4...
    # (Insert existing evaluation calls here)

    # 5. Latency
    print("\n[5/5] Measuring HNSW Latency ...")
    lat = benchmark_hnsw_latency()
    print(f"      Avg Query Latency: {lat['avg_query_ms']} ms")

    # Save results
    results_path = os.path.join(args.output, "results.json")
    # (Write to json logic remains same)
    print(f"\n✅ All results saved → {results_path}")

if __name__ == "__main__":
    main()