"""
merge.py — Local label-merge logic for FaceVault federated sync.

Consumes merge suggestions from sync_server.py and updates the local
FaceVault database (app_data.pkl) without touching images.
"""

import os
import pickle
import shutil
from typing import Dict, List, Tuple

from config import DATA_FILE, INDEX_FILE, IMAGES_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Preview (non-destructive)
# ──────────────────────────────────────────────────────────────────────────────

def preview_merges(
    suggestions: List[dict],
    clusters: Dict[int, List[str]],
    device_id: str,
) -> List[dict]:
    """Return a human-readable summary of proposed merges.

    Each suggestion dict from the server looks like:
        {device_a, cluster_a, device_b, cluster_b, similarity}

    Returns a list of dicts:
        {local_cluster_id, local_label, remote_device, remote_cluster_id, similarity}
    """
    previews = []
    for s in suggestions:
        # Determine which cluster is ours
        if s["device_a"] == device_id:
            local_cid = s["cluster_a"]
            remote_dev = s["device_b"]
            remote_cid = s["cluster_b"]
        else:
            local_cid = s["cluster_b"]
            remote_dev = s["device_a"]
            remote_cid = s["cluster_a"]

        # Look up the local label from the first image path's parent folder
        local_label = _label_for_cluster(local_cid, clusters)

        previews.append({
            "local_cluster_id": local_cid,
            "local_label": local_label,
            "remote_device": remote_dev,
            "remote_cluster_id": remote_cid,
            "similarity": s["similarity"],
        })
    return previews


def _label_for_cluster(cluster_id: int, clusters: Dict[int, List[str]]) -> str:
    """Extract the human-readable label from a cluster's first image path."""
    paths = clusters.get(cluster_id, [])
    if not paths:
        return f"Unknown (ID {cluster_id})"
    return os.path.basename(os.path.dirname(paths[0])).replace("_", " ")


# ──────────────────────────────────────────────────────────────────────────────
# Apply merges
# ──────────────────────────────────────────────────────────────────────────────

def apply_merge_suggestions(
    suggestions: List[dict],
    clusters: Dict[int, List[str]],
    paths: List[str],
    device_id: str,
    new_labels: Dict[int, str] | None = None,
) -> Tuple[Dict[int, List[str]], List[str], int]:
    """Apply merge suggestions to the local database.

    Parameters
    ----------
    suggestions : list of match dicts from the server
    clusters    : current cluster dict (mutated in-place)
    paths       : current paths list (may be mutated)
    device_id   : this device's ID
    new_labels  : optional {local_cluster_id: new_name} overrides

    Returns
    -------
    (updated_clusters, updated_paths, merge_count)

    The merge renames the local cluster's image folder to match the
    agreed-upon label.  This keeps images on disk consistent with the
    cluster metadata.
    """
    if new_labels is None:
        new_labels = {}

    merge_count = 0
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGES_DIR)

    for s in suggestions:
        if s["device_a"] == device_id:
            local_cid = s["cluster_a"]
        else:
            local_cid = s["cluster_b"]

        if local_cid not in clusters:
            continue

        # Determine the new label (user override or keep current)
        current_label = _label_for_cluster(local_cid, clusters)
        target_label = new_labels.get(local_cid, current_label)

        if target_label == current_label:
            # No rename needed — just mark as synced
            merge_count += 1
            continue

        # Rename the image folder (with backup for rollback safety)
        old_dir = os.path.join(base_dir, current_label.replace(" ", "_"))
        new_dir = os.path.join(base_dir, target_label.replace(" ", "_"))
        backup_dir = old_dir + "_backup"

        try:
            # Create a backup before any destructive operation
            if os.path.exists(old_dir):
                shutil.copytree(old_dir, backup_dir)

            if os.path.exists(old_dir) and not os.path.exists(new_dir):
                os.rename(old_dir, new_dir)
            elif os.path.exists(old_dir) and os.path.exists(new_dir):
                # Both folders exist — move contents into target
                for fname in os.listdir(old_dir):
                    shutil.move(os.path.join(old_dir, fname), new_dir)
                os.rmdir(old_dir)

            # Remove backup on success
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)

        except Exception:
            # Rollback: restore from backup if the rename/move failed
            if os.path.exists(backup_dir):
                if os.path.exists(old_dir):
                    shutil.rmtree(old_dir)
                os.rename(backup_dir, old_dir)
            raise  # re-raise so the caller knows it failed

        # Update paths in cluster
        updated = []
        for p in clusters[local_cid]:
            fname = os.path.basename(p)
            updated.append(os.path.join(new_dir, fname))
        clusters[local_cid] = updated

        # Update global paths list too
        for i, p in enumerate(paths):
            if os.path.dirname(p) == old_dir:
                paths[i] = os.path.join(new_dir, os.path.basename(p))

        merge_count += 1

    return clusters, paths, merge_count


def save_merged_state(clusters: Dict[int, List[str]], paths: List[str]) -> None:
    """Persist updated clusters and paths back to disk."""
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"clusters": clusters, "paths": paths}, f)
