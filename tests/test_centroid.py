"""Tests for centroid.py — centroid computation, EMA, fragmentation merge."""

import numpy as np
import faiss
import pytest
from centroid import (
    compute_centroids,
    compute_centroids_ema,
    merge_fragmented_clusters,
    _get_vectors_for_cluster,
)
from config import EMBEDDING_DIM


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_index_and_clusters(n_clusters=3, imgs_per_cluster=5, dim=EMBEDDING_DIM):
    """Create a synthetic FAISS index with known clusters."""
    index = faiss.IndexFlatIP(dim)
    paths = []
    clusters = {}
    for cid in range(n_clusters):
        clist = []
        base = np.random.randn(dim).astype("float32")
        base /= np.linalg.norm(base)
        for j in range(imgs_per_cluster):
            vec = base + np.random.randn(dim).astype("float32") * 0.05
            vec /= np.linalg.norm(vec)
            index.add(vec.reshape(1, -1))
            p = f"/fake/{cid}/img_{j}.jpg"
            paths.append(p)
            clist.append(p)
        clusters[cid] = clist
    return clusters, paths, index


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestComputeCentroids:
    def test_output_shape_and_norm(self):
        clusters, paths, index = _make_index_and_clusters()
        centroids = compute_centroids(clusters, paths, index)
        assert len(centroids) == 3
        for vec in centroids.values():
            assert vec.shape == (EMBEDDING_DIM,)
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-4, "Centroid should be L2-normalised"

    def test_skips_empty_and_noise(self):
        clusters, paths, index = _make_index_and_clusters()
        clusters[-1] = []  # noise cluster
        centroids = compute_centroids(clusters, paths, index)
        assert -1 not in centroids

    def test_single_image_cluster(self):
        clusters, paths, index = _make_index_and_clusters(n_clusters=1, imgs_per_cluster=1)
        centroids = compute_centroids(clusters, paths, index)
        assert len(centroids) == 1


class TestEMA:
    def test_ema_moves_toward_new_batch(self):
        clusters, paths, index = _make_index_and_clusters(n_clusters=1, imgs_per_cluster=5)
        prev = compute_centroids(clusters, paths, index)

        # Add more vectors (simulate batch 2)
        dim = EMBEDDING_DIM
        new_base = np.random.randn(dim).astype("float32")
        new_base /= np.linalg.norm(new_base)
        for j in range(5):
            vec = new_base + np.random.randn(dim).astype("float32") * 0.05
            vec /= np.linalg.norm(vec)
            index.add(vec.reshape(1, -1))
            p = f"/fake/0/img_new_{j}.jpg"
            paths.append(p)
            clusters[0].append(p)

        ema = compute_centroids_ema(clusters, paths, index, prev_centroids=prev, alpha=0.5)
        simple = compute_centroids(clusters, paths, index)

        # EMA centroid should differ from simple mean
        assert ema[0].shape == (EMBEDDING_DIM,)
        assert not np.allclose(ema[0], simple[0], atol=1e-3)

    def test_no_prev_equals_simple_mean(self):
        clusters, paths, index = _make_index_and_clusters()
        ema = compute_centroids_ema(clusters, paths, index, prev_centroids=None)
        simple = compute_centroids(clusters, paths, index)
        for cid in simple:
            assert np.allclose(ema[cid], simple[cid], atol=1e-5)


class TestMergeFragmented:
    def test_identical_centroids_merged(self):
        vec = np.random.randn(EMBEDDING_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        centroids = {0: vec.copy(), 1: vec.copy()}
        merged, merge_map = merge_fragmented_clusters(centroids, threshold=0.99)
        # One should be absorbed
        assert len(merged) == 1
        assert len(merge_map) == 1

    def test_distant_centroids_not_merged(self):
        v1 = np.random.randn(EMBEDDING_DIM).astype("float32"); v1 /= np.linalg.norm(v1)
        v2 = -v1  # maximally distant
        centroids = {0: v1, 1: v2}
        merged, merge_map = merge_fragmented_clusters(centroids, threshold=0.9)
        assert len(merged) == 2
        assert len(merge_map) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
