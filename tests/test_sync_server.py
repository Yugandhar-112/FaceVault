"""Tests for sync_server.py — Flask relay routes using test client."""

import json
import numpy as np
import pytest
from cryptography.fernet import Fernet

from sync_server import app as flask_app
from privacy import generate_sync_manifest, manifest_to_json
from config import EMBEDDING_DIM, SYNC_MATCH_THRESHOLD


def _random_centroid(dim=EMBEDDING_DIM):
    v = np.random.randn(dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        # Reset server state before each test
        c.post("/reset")
        yield c


class TestSubmitManifest:
    def test_submit_single_device(self, client):
        centroids = {0: _random_centroid()}
        manifest = generate_sync_manifest("dev_A", centroids, epsilon=5.0)
        resp = client.post(
            "/submit_manifest",
            data=manifest_to_json(manifest),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["clusters_received"] == 1
        assert data["devices_registered"] == 1

    def test_submit_two_devices(self, client):
        for dev_id in ("dev_A", "dev_B"):
            centroids = {i: _random_centroid() for i in range(3)}
            m = generate_sync_manifest(dev_id, centroids, epsilon=5.0)
            resp = client.post("/submit_manifest", data=manifest_to_json(m), content_type="application/json")
            assert resp.status_code == 200
        # Check count
        data = resp.get_json()
        assert data["devices_registered"] == 2


class TestCompare:
    def test_needs_two_devices(self, client):
        resp = client.get("/compare")
        data = resp.get_json()
        assert data["status"] == "waiting"

    def test_matching_centroids_found(self, client):
        # Both devices share the SAME centroid for cluster 0 → should match
        # Use extremely high ε and large cluster_size to make DP noise negligible
        shared = _random_centroid()
        for dev_id in ("dev_A", "dev_B"):
            centroids = {0: shared.copy()}
            sizes = {0: 100}  # large cluster → lower sensitivity
            m = generate_sync_manifest(dev_id, centroids, cluster_sizes=sizes, epsilon=1000.0)
            client.post("/submit_manifest", data=manifest_to_json(m), content_type="application/json")

        resp = client.get("/compare")
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["total_matches"] >= 1
        # The match should involve cluster 0 from both devices
        match = data["matches"][0]
        assert match["cluster_a"] == 0
        assert match["cluster_b"] == 0
        assert match["similarity"] >= SYNC_MATCH_THRESHOLD

    def test_unrelated_centroids_no_match(self, client):
        c1 = _random_centroid()
        c2 = -c1  # opposite direction
        m1 = generate_sync_manifest("dev_A", {0: c1}, epsilon=100.0)
        m2 = generate_sync_manifest("dev_B", {0: c2}, epsilon=100.0)
        client.post("/submit_manifest", data=manifest_to_json(m1), content_type="application/json")
        client.post("/submit_manifest", data=manifest_to_json(m2), content_type="application/json")

        resp = client.get("/compare")
        data = resp.get_json()
        assert data["total_matches"] == 0


class TestMergeSuggestions:
    def test_no_suggestions_initially(self, client):
        resp = client.get("/merge_suggestions/dev_A")
        data = resp.get_json()
        assert data["suggestions"] == []

    def test_suggestions_after_compare(self, client):
        shared = _random_centroid()
        for dev_id in ("dev_A", "dev_B"):
            sizes = {0: 100}
            m = generate_sync_manifest(dev_id, {0: shared.copy()}, cluster_sizes=sizes, epsilon=1000.0)
            client.post("/submit_manifest", data=manifest_to_json(m), content_type="application/json")
        client.get("/compare")

        resp = client.get("/merge_suggestions/dev_A")
        data = resp.get_json()
        assert len(data["suggestions"]) >= 1


class TestStatusAndReset:
    def test_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_reset_clears_state(self, client):
        m = generate_sync_manifest("dev_A", {0: _random_centroid()}, epsilon=5.0)
        client.post("/submit_manifest", data=manifest_to_json(m), content_type="application/json")
        client.post("/reset")
        resp = client.get("/status")
        data = resp.get_json()
        assert data["devices_registered"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
