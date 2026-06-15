"""
centroid.py — Per-cluster centroid computation for FaceVault.

Computes, maintains, and pre-processes L2-normalised centroids
from the existing FAISS index and cluster mappings.
"""

import numpy as np
import faiss
import pickle
import os
from typing import Dict, List, Optional, Tuple

from config import (
    EMBEDDING_DIM,
    DATA_FILE,
    INDEX_FILE,
    CENTROID_EMA_ALPHA,
    CENTROID_MERGE_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_vectors_for_cluster(
    cluster_paths: List[str],
    all_paths: List[str],
    index: faiss.Index,
) -> np.ndarray:
    """Return the FAISS vectors that belong to *cluster_paths*.

    Looks up each path's position in *all_paths* and reconstructs the
    corresponding vector from the FAISS index.
    """
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    ids = [path_to_idx[p] for p in cluster_paths if p in path_to_idx]
    if not ids:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")
    vectors = np.vstack([index.reconstruct(i) for i in ids]).astype("float32")
    return vectors


# ──────────────────────────────────────────────────────────────────────────────
# Core API
# ──────────────────────────────────────────────────────────────────────────────

def compute_centroids(
    clusters: Dict[int, List[str]],
    paths: List[str],
    index: faiss.Index,
) -> Dict[int, np.ndarray]:
    """Compute the mean centroid for every cluster.

    Returns
    -------
    dict  {cluster_id: L2-normalised centroid (512-d float32)}
    """
    centroids: Dict[int, np.ndarray] = {}
    for c_id, c_paths in clusters.items():
        if c_id == -1 or len(c_paths) == 0:
            continue
        vectors = _get_vectors_for_cluster(c_paths, paths, index)
        if vectors.shape[0] == 0:
            continue
        mean = vectors.mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(mean)
        centroids[c_id] = mean.squeeze()
    return centroids


def compute_centroids_ema(
    clusters: Dict[int, List[str]],
    paths: List[str],
    index: faiss.Index,
    prev_centroids: Optional[Dict[int, np.ndarray]] = None,
    alpha: float = CENTROID_EMA_ALPHA,
) -> Dict[int, np.ndarray]:
    """Exponential-moving-average centroid update.

    c_new = α · c_batch + (1 − α) · c_old

    This dampens centroid drift when photos are added incrementally.
    Falls back to a simple mean when no previous centroid exists for a cluster.
    """
    batch_centroids = compute_centroids(clusters, paths, index)

    if prev_centroids is None:
        return batch_centroids

    ema_centroids: Dict[int, np.ndarray] = {}
    for c_id, c_batch in batch_centroids.items():
        if c_id in prev_centroids:
            c_old = prev_centroids[c_id]
            c_new = (alpha * c_batch + (1.0 - alpha) * c_old).astype("float32")
            c_new = c_new.reshape(1, -1)
            faiss.normalize_L2(c_new)
            ema_centroids[c_id] = c_new.squeeze()
        else:
            ema_centroids[c_id] = c_batch
    return ema_centroids


def merge_fragmented_clusters(
    centroids: Dict[int, np.ndarray],
    threshold: float = CENTROID_MERGE_THRESHOLD,
    cluster_sizes: Optional[Dict[int, int]] = None,
    min_cluster_size: int = 3,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Merge local clusters whose centroids are suspiciously close.

    Parameters
    ----------
    centroids        : {cluster_id: centroid_vector}
    threshold        : cosine-similarity above which clusters are merged
    cluster_sizes    : {cluster_id: n_images} — used for the min-size guard
    min_cluster_size : clusters smaller than this are excluded from merging
                       to avoid collapsing genuinely different people
                       (e.g. twins, look-alikes with few photos).

    Returns
    -------
    merged_centroids : dict — new centroid map after merging
    merge_map        : dict — {absorbed_id: surviving_id} for bookkeeping

    Known edge case
    ---------------
    Twins or look-alikes with similar embeddings *and* enough photos to
    exceed `min_cluster_size` will still be merged incorrectly. This is
    acknowledged as a limitation — a human-review step before export is
    recommended for high-stakes deployments.
    """
    ids = sorted(centroids.keys())
    if len(ids) < 2:
        return dict(centroids), {}

    # Build a matrix of all centroids and compute pairwise cosine similarity
    mat = np.vstack([centroids[i] for i in ids]).astype("float32")
    faiss.normalize_L2(mat)
    sim = mat @ mat.T                       # cosine similarity matrix

    # Union-Find for cluster merging
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only consider clusters that meet the minimum size requirement
    if cluster_sizes is not None:
        eligible = {cid for cid in ids if cluster_sizes.get(cid, 0) >= min_cluster_size}
    else:
        eligible = set(ids)  # if sizes unknown, allow all

    for i_pos, id_a in enumerate(ids):
        for j_pos in range(i_pos + 1, len(ids)):
            id_b = ids[j_pos]
            if id_a in eligible and id_b in eligible and sim[i_pos, j_pos] >= threshold:
                union(id_a, id_b)

    # Group clusters by their root
    groups: Dict[int, List[int]] = {}
    for cid in ids:
        root = find(cid)
        groups.setdefault(root, []).append(cid)

    # Rebuild centroids and merge_map
    merged_centroids: Dict[int, np.ndarray] = {}
    merge_map: Dict[int, int] = {}
    for root, members in groups.items():
        # Surviving centroid = mean of all members' centroids
        vecs = np.vstack([centroids[m] for m in members]).astype("float32")
        mean = vecs.mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(mean)
        merged_centroids[root] = mean.squeeze()
        for m in members:
            if m != root:
                merge_map[m] = root

    return merged_centroids, merge_map


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────

def load_and_compute() -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]], List[str]]:
    """Load FaceVault data from disk and compute centroids.

    Returns (centroids, clusters, paths).
    """
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
    clusters = data["clusters"]
    paths = data["paths"]
    centroids = compute_centroids(clusters, paths, index)
    return centroids, clusters, paths
