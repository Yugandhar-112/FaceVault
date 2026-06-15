"""Tests for merge.py — preview and apply merge suggestions."""

import os
import pytest
from merge import preview_merges, apply_merge_suggestions, _label_for_cluster


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_clusters():
    return {
        0: ["/images/Alice/img_001.jpg", "/images/Alice/img_002.jpg"],
        1: ["/images/Bob/img_001.jpg"],
        2: ["/images/Charlie/img_001.jpg", "/images/Charlie/img_002.jpg"],
    }


@pytest.fixture
def sample_paths():
    return [
        "/images/Alice/img_001.jpg",
        "/images/Alice/img_002.jpg",
        "/images/Bob/img_001.jpg",
        "/images/Charlie/img_001.jpg",
        "/images/Charlie/img_002.jpg",
    ]


@pytest.fixture
def sample_suggestions():
    return [
        {
            "device_a": "dev_A",
            "cluster_a": 0,
            "device_b": "dev_B",
            "cluster_b": 5,
            "similarity": 0.92,
        },
        {
            "device_a": "dev_B",
            "cluster_a": 7,
            "device_b": "dev_A",
            "cluster_b": 1,
            "similarity": 0.85,
        },
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLabelForCluster:
    def test_extracts_name(self, sample_clusters):
        assert _label_for_cluster(0, sample_clusters) == "Alice"
        assert _label_for_cluster(1, sample_clusters) == "Bob"

    def test_unknown_for_missing(self, sample_clusters):
        label = _label_for_cluster(99, sample_clusters)
        assert "Unknown" in label


class TestPreviewMerges:
    def test_returns_correct_count(self, sample_suggestions, sample_clusters):
        previews = preview_merges(sample_suggestions, sample_clusters, "dev_A")
        assert len(previews) == 2

    def test_identifies_local_cluster(self, sample_suggestions, sample_clusters):
        previews = preview_merges(sample_suggestions, sample_clusters, "dev_A")
        local_ids = {p["local_cluster_id"] for p in previews}
        assert 0 in local_ids
        assert 1 in local_ids

    def test_contains_similarity(self, sample_suggestions, sample_clusters):
        previews = preview_merges(sample_suggestions, sample_clusters, "dev_A")
        for p in previews:
            assert "similarity" in p
            assert 0 < p["similarity"] <= 1.0


class TestApplyMergeSuggestions:
    def test_returns_merge_count(self, sample_suggestions, sample_clusters, sample_paths):
        # No label overrides → count should still increment (no-op renames)
        clusters, paths, count = apply_merge_suggestions(
            sample_suggestions, sample_clusters, sample_paths, "dev_A"
        )
        assert count == 2

    def test_no_suggestions_no_change(self, sample_clusters, sample_paths):
        clusters, paths, count = apply_merge_suggestions(
            [], sample_clusters, sample_paths, "dev_A"
        )
        assert count == 0
        assert clusters == sample_clusters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
