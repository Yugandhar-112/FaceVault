# FaceVault 🔐

> **Dynamic, unsupervised face identity management** with **privacy-preserving federated sync**.  
> Powered by **ArcFace** embeddings + **FAISS-HNSW** search + **Differential Privacy** + **Fernet encryption**.

---

## Overview

FaceVault is a Streamlit web application that lets you build, search, and **collaboratively sync** a face identity database **on the fly**. It uses state-of-the-art metric learning instead of traditional softmax classification, which means:

- **New identities can be registered instantly** — just upload a photo and give them a name.
- **No GPU or retraining needed** — deep features are extracted once per image; all similarity logic runs on CPU in milliseconds.
- **Portable** — the entire database fits in two files (`app_data.pkl` + `vector_index.bin`).
- **Privacy-first sync** — two devices can collaboratively label shared identities by exchanging only **encrypted, DP-noised centroids**, never raw images.

---

## Architecture

```
 Device A                                    Device B
┌──────────────┐                       ┌──────────────┐
│ ArcFace CNN  │                       │ ArcFace CNN  │
│ (512-d emb.) │                       │ (512-d emb.) │
└──────┬───────┘                       └──────┬───────┘
       ▼                                      ▼
┌──────────────┐                       ┌──────────────┐
│ FAISS HNSW   │                       │ FAISS HNSW   │
│ O(log n)     │                       │ O(log n)     │
└──────┬───────┘                       └──────┬───────┘
       ▼                                      ▼
┌──────────────┐                       ┌──────────────┐
│ Cluster Dict │                       │ Cluster Dict │
│ + EMA Centr. │                       │ + EMA Centr. │
└──────┬───────┘                       └──────┬───────┘
       │  DP noise + Fernet encrypt           │
       ▼                                      ▼
   ╔══════════════════════════════════════════════╗
   ║   Sync Server (Flask blind relay)            ║
   ║   • Decrypts centroids for cosine compare    ║
   ║   • Never sees/stores raw images             ║
   ║   • Returns merge suggestions per device     ║
   ╚══════════════════════════════════════════════╝
       │              │                     │
       ▼              ▼                     ▼
  Merge suggestions   Merge suggestions
  applied locally     applied locally
```

### Core Modules

| Module | Technology | Purpose |
|--------|-----------|---------|
| Face embedding | ArcFace (via `deepface`) | 512-dim angular-margin features |
| Similarity search | FAISS HNSW (`faiss-cpu`) | Sub-linear nearest-neighbour lookup |
| Centroid computation | `centroid.py` | EMA-smoothed centroids + fragmentation merge |
| Privacy layer | `privacy.py` | Laplacian DP noise (ε-DP) + Fernet AES-128 encryption |
| Sync relay | `sync_server.py` (Flask) | Blind relay for cross-device centroid comparison |
| Merge logic | `merge.py` | Local label merging with backup & rollback |
| Benchmarks | `benchmark.py` | P/R/F1, ε-accuracy curves, centroid stability, encryption timing |
| State persistence | `pickle` + `faiss.write_index` | Survive Streamlit re-runs |
| UI | Streamlit | Browser-based interface with 3 tabs |

---

## Features

| Tab | Description |
|-----|-------------|
| 🔍 **Database Explorer** | Browse all registered identities; filter by name or cluster ID. |
| ➕ **Add / Search Face** | Upload a photo → ArcFace extract → HNSW search → add to existing person **or** register as new. Includes deduplication guard. |
| 🔄 **Federated Sync** | Compute EMA centroids → apply DP noise → encrypt → submit to relay server → compare cross-device → preview & apply merge suggestions. |

`app_nonHNSW.py` provides a lightweight read-only explorer using a flat (brute-force) FAISS index — useful for debugging or comparing search quality.

---

## Federated Cluster Sync — How It Works

1. **Centroid Computation** — Each device computes L2-normalised centroids for its clusters using an **Exponential Moving Average** (α = 0.3) to combat centroid drift from incremental photo additions. Nearby clusters are pre-merged (union-find, threshold 0.80) with a minimum cluster size guard to avoid collapsing genuinely different people (twins/look-alikes).

2. **Differential Privacy** — Before export, calibrated **Laplacian noise** is added to each centroid. Sensitivity = `2/n` (unit-normalised vectors); scale = `sensitivity / ε`. The centroid is re-normalised after noise addition. Lower ε → more private, noisier output.

3. **Encryption** — Noised centroids are serialised via `msgpack` and encrypted with **Fernet** (AES-128-CBC). The ciphertext + key are bundled into a `SyncManifest`.

4. **Relay Comparison** — The Flask sync server receives manifests from multiple devices, decrypts centroids (semi-honest model), and computes pairwise **cosine similarity** across devices. Pairs exceeding the sync threshold (0.70) are flagged as merge suggestions.

5. **Local Merge** — Each device fetches its suggestions, previews them (showing local label, remote device, similarity score), and applies merges locally. Folder renames are backed up with `copytree` for rollback safety.

### Privacy Model

> **⚠️ Semi-honest relay.** In this demo the key travels with the ciphertext. The server decrypts for comparison but is trusted not to leak plaintext centroids. A production deployment would use **homomorphic encryption** (CKKS / TenSEAL) or **SMPC**. This limitation is explicitly surfaced in the app UI (Tab 3 disclaimer).

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/FaceVault.git
cd FaceVault
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the image dataset

FaceVault expects face images organised as:

```
images/
├── Person_Name/
│   ├── Person_Name_0001.jpg
│   └── …
└── Another_Person/
    └── …
```

> **Recommended dataset:** [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)  
> Extract it into the `images/` directory — the path-fixer in `load_data()` normalises paths automatically across machines.

### 3. Build the initial FAISS index

If you already have `app_data.pkl` and `vector_index.bin` (e.g. shared by a collaborator), place them in the project root and skip this step.

Otherwise, run the provided notebook (or your own indexing script) to populate the files from scratch.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Run the federated sync demo (two-device simulation)

```bash
# Terminal 1 — Sync relay server
python sync_server.py

# Terminal 2 — Device A
streamlit run app.py --server.port 8501

# Terminal 3 — Device B
streamlit run app.py --server.port 8502
```

1. On each device, go to the **🔄 Federated Sync** tab.
2. Set a unique **Device ID** (e.g. `device_A`, `device_B`).
3. Click **🚀 Export & Submit** on both devices.
4. Click **🔀 Run Cross-Device Comparison** on either.
5. Click **📥 Fetch My Suggestions** to see matching identities.
6. Click **✅ Apply Merges** to sync labels.

### 6. Run benchmarks

```bash
python benchmark.py --output benchmarks/
```

Produces `benchmarks/results.json`, `epsilon_accuracy.png`, and `centroid_stability.png`.

### 7. Run tests

```bash
python -m pytest tests/ -v
```

32 tests covering centroid computation, DP noise, encryption, server routes, and merge logic.

---

## Configuration

All tuneable parameters live in **`config.py`**:

| Constant | Default | Description |
|----------|---------|-------------|
| `MATCH_THRESHOLD` | `0.50` | Cosine-similarity for confident match |
| `DEDUP_THRESHOLD` | `0.85` | Warn user of likely duplicate |
| `HNSW_M` | `32` | HNSW graph connectivity (higher → better recall, more RAM) |
| `HNSW_EF_SEARCH` | `128` | Query-time beam width (higher → better recall, slower) |
| `EMBEDDING_DIM` | `512` | ArcFace embedding dimension (fixed) |
| `THUMBNAIL_COLUMNS` | `5` | Gallery columns in explorer view |
| `CENTROID_EMA_ALPHA` | `0.3` | EMA smoothing for centroid updates |
| `CENTROID_MERGE_THRESHOLD` | `0.80` | Merge fragmented local clusters before export |
| `DP_EPSILON` | `1.0` | Differential privacy budget (lower = more private) |
| `SYNC_MATCH_THRESHOLD` | `0.70` | Cosine-similarity for cross-device merge suggestion |
| `SYNC_SERVER_HOST` | `127.0.0.1` | Relay server bind address |
| `SYNC_SERVER_PORT` | `5050` | Relay server port |

---

## Project Structure

```
FaceVault/
├── app.py                # Main Streamlit app (3 tabs: Explorer, Add/Search, Federated Sync)
├── app_nonHNSW.py        # Read-only explorer (flat FAISS index)
├── config.py             # All tuneable constants (matching, HNSW, sync, DP)
├── centroid.py            # EMA centroid computation + fragmentation merge
├── privacy.py             # Differential privacy noise + Fernet encryption
├── sync_server.py         # Flask blind relay for cross-device comparison
├── merge.py               # Local merge logic with backup & rollback
├── benchmark.py           # Evaluation suite: P/R/F1, ε-accuracy, stability, timing
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes large binaries and dataset
├── README.md
└── tests/
    ├── __init__.py
    ├── test_centroid.py    # 7 tests — centroid computation, EMA, fragmentation merge
    ├── test_privacy.py     # 9 tests — DP noise, encryption round-trip, manifests
    ├── test_sync_server.py # 9 tests — Flask routes, comparison, suggestions
    └── test_merge.py       # 7 tests — label extraction, preview, apply merges
```

---

## Notes & Limitations

- `vector_index.bin` and `app_data.pkl` are excluded from version control (see `.gitignore`) because they can be several hundred MB.  
  Share them separately (e.g. Google Drive, Git LFS, DVC) or regenerate from the dataset.
- The LFW image dataset is similarly excluded. Download it directly from the [official source](http://vis-www.cs.umass.edu/lfw/).
- ArcFace embeddings are extracted with CPU inference; expect ~1–2 s per image on a modern laptop.
- **Known edge case:** Twins or look-alikes whose embeddings are very similar may be incorrectly merged during federated sync. A minimum cluster-size guard (default: 3 images) reduces this risk but does not eliminate it. A human-review step before export is recommended for high-stakes use.
- **HNSW `reconstruct()`:** The HNSW index does not support `reconstruct()` by default. The benchmark suite uses a synthetic `IndexFlatIP` for evaluation. To adapt benchmarks for a real HNSW index, either store raw embeddings separately or create the index with `store_pairs=True`.

---

## License

MIT — see `LICENSE` for details.
