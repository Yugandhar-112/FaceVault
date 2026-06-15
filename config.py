"""
config.py — Centralised configuration for FaceVault.

Edit these constants to tune the application's behaviour
without touching the main application code.
"""

# ── Matching ──────────────────────────────────────────────────────────────────
# Cosine-similarity threshold (L2-normalised vectors → L2 distance ≈ cosine distance).
# A query whose nearest-neighbour distance is *above* this value is considered
# a confident match; below it the match is flagged as low-confidence.
#
# Acceptable range: 0.0 – 2.0  (typical safe values: 0.40 – 0.65)
MATCH_THRESHOLD: float = 0.50

# Deduplication threshold — if the nearest-neighbour score exceeds this value
# when adding a new face, warn the user that it may be a duplicate.
# Should be ≥ MATCH_THRESHOLD.  Set to 2.0 to effectively disable dedup warnings.
DEDUP_THRESHOLD: float = 0.85

# ── FAISS HNSW index parameters ───────────────────────────────────────────────
# M      : number of bi-directional links per node (higher → better recall, more RAM)
# efSearch: beam width at query time (higher → better recall, slower)
HNSW_M: int = 32
HNSW_EF_SEARCH: int = 128

# ArcFace embedding dimension (fixed by the model — do not change)
EMBEDDING_DIM: int = 512

# ── File paths ────────────────────────────────────────────────────────────────
DATA_FILE: str = "app_data.pkl"
INDEX_FILE: str = "vector_index.bin"           # HNSW index (used by app.py)
FLAT_INDEX_FILE: str = "vector_index_flat.bin"  # Flat index (used by app_nonHNSW.py)
IMAGES_DIR: str = "images"

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE: str = "FaceVault"
PAGE_ICON: str = "🔐"
THUMBNAIL_COLUMNS: int = 5          # number of image columns in the gallery view

# ── Federated Sync ────────────────────────────────────────────────────────────
CENTROID_EMA_ALPHA: float = 0.3          # EMA smoothing for centroid updates
CENTROID_MERGE_THRESHOLD: float = 0.80   # merge fragmented local clusters before export
DP_EPSILON: float = 1.0                  # differential privacy budget (lower = more private)
SYNC_MATCH_THRESHOLD: float = 0.70       # cosine sim threshold for cross-device merge suggestion
SYNC_SERVER_HOST: str = "127.0.0.1"
SYNC_SERVER_PORT: int = 5050
MANIFEST_FILE: str = "sync_manifest.enc"  # exported manifest filename
