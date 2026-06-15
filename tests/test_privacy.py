"""Tests for privacy.py — DP noise, encryption round-trip, manifest generation."""

import numpy as np
import pytest
from cryptography.fernet import Fernet

from privacy import (
    add_dp_noise,
    encrypt_manifest,
    decrypt_manifest,
    generate_sync_manifest,
    manifest_to_json,
    manifest_from_json,
    SyncManifest,
)
from config import EMBEDDING_DIM


def _random_centroid(dim=EMBEDDING_DIM):
    v = np.random.randn(dim).astype("float32")
    return v / np.linalg.norm(v)


class TestDPNoise:
    def test_output_shape(self):
        c = _random_centroid()
        noisy = add_dp_noise(c, epsilon=1.0, cluster_size=10)
        assert noisy.shape == c.shape

    def test_output_is_unit_length(self):
        c = _random_centroid()
        noisy = add_dp_noise(c, epsilon=1.0, cluster_size=5)
        assert abs(np.linalg.norm(noisy) - 1.0) < 1e-5

    def test_output_differs_from_input(self):
        c = _random_centroid()
        noisy = add_dp_noise(c, epsilon=1.0, cluster_size=5)
        assert not np.allclose(c, noisy, atol=1e-6)

    def test_high_epsilon_preserves_similarity(self):
        c = _random_centroid()
        noisy = add_dp_noise(c, epsilon=100.0, cluster_size=50)
        sim = float(np.dot(c, noisy))
        assert sim > 0.95, f"High ε should preserve direction, got sim={sim}"

    def test_low_epsilon_adds_more_noise(self):
        c = _random_centroid()
        sims = []
        for _ in range(50):
            noisy = add_dp_noise(c, epsilon=0.1, cluster_size=1)
            sims.append(float(np.dot(c, noisy)))
        avg_sim = np.mean(sims)
        # Low ε with cluster_size=1 should add significant noise
        assert avg_sim < 0.95


class TestEncryptDecrypt:
    def test_round_trip(self):
        centroids = {0: _random_centroid(), 1: _random_centroid()}
        key = Fernet.generate_key()
        ct = encrypt_manifest(centroids, key)
        recovered = decrypt_manifest(ct, key)
        for cid in centroids:
            assert np.allclose(centroids[cid], recovered[cid], atol=1e-6)

    def test_wrong_key_fails(self):
        centroids = {0: _random_centroid()}
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        ct = encrypt_manifest(centroids, key1)
        with pytest.raises(Exception):
            decrypt_manifest(ct, key2)


class TestSyncManifest:
    def test_generate_produces_valid_manifest(self):
        centroids = {0: _random_centroid(), 1: _random_centroid(), 2: _random_centroid()}
        manifest = generate_sync_manifest("test_device", centroids, epsilon=1.0)
        assert isinstance(manifest, SyncManifest)
        assert manifest.device_id == "test_device"
        assert len(manifest.cluster_ids) == 3
        assert isinstance(manifest.ciphertext, bytes)
        assert manifest.key  # not empty

    def test_manifest_json_round_trip(self):
        centroids = {5: _random_centroid()}
        manifest = generate_sync_manifest("dev_A", centroids)
        json_str = manifest_to_json(manifest)
        recovered = manifest_from_json(json_str)
        assert recovered.device_id == "dev_A"
        assert recovered.cluster_ids == manifest.cluster_ids
        # Decrypt and verify
        key = recovered.key.encode("utf-8")
        dec = decrypt_manifest(recovered.ciphertext, key)
        assert 5 in dec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
