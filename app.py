import streamlit as st
import pickle
import faiss
import numpy as np
from deepface import DeepFace
import cv2
import os
import tempfile
import time
import requests
import json

from config import (
    MATCH_THRESHOLD,
    DEDUP_THRESHOLD,
    DATA_FILE,
    INDEX_FILE,
    IMAGES_DIR,
    APP_TITLE,
    PAGE_ICON,
    THUMBNAIL_COLUMNS,
    DP_EPSILON,
    SYNC_SERVER_HOST,
    SYNC_SERVER_PORT,
    CENTROID_EMA_ALPHA,
)
from centroid import compute_centroids, compute_centroids_ema, merge_fragmented_clusters
from privacy import generate_sync_manifest, manifest_to_json, manifest_from_json
from merge import preview_merges, apply_merge_suggestions, save_merged_state

# --- 1. CONFIGURATION ---
st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} {APP_TITLE}")
st.caption("Dynamic identity clustering powered by ArcFace + FAISS-HNSW")

# MATCH_THRESHOLD is imported from config.py

# --- 2. DATA MANAGEMENT FUNCTIONS ---

def load_data():
    """Loads the database and fixes paths for the local machine."""
    if not os.path.exists(DATA_FILE) or not os.path.exists(INDEX_FILE):
        return {}, [], None

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
    
    # Path Fixer Logic
    old_paths = data['paths']
    new_paths = []
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    
    # Create images folder if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Fix paths in the list
    for p in old_paths:
        filename = os.path.basename(p)
        parent_folder = os.path.basename(os.path.dirname(p))
        local_path = os.path.join(base_dir, parent_folder, filename)
        new_paths.append(local_path)
        
    # Fix paths in the clusters dict
    new_clusters = {}
    for c_id, p_list in data['clusters'].items():
        fixed_list = []
        for p in p_list:
            fname = os.path.basename(p)
            parent = os.path.basename(os.path.dirname(p))
            fixed_list.append(os.path.join(base_dir, parent, fname))
        new_clusters[c_id] = fixed_list
            
    return new_clusters, new_paths, index

def save_database(clusters, paths, index):
    """Saves the updated clusters, paths, and vector index to disk."""
    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'clusters': clusters, 'paths': paths}, f)
    faiss.write_index(index, INDEX_FILE)

def add_new_face(img_array, embedding, person_name, cluster_id, clusters, paths, index):
    """Adds a new face to the disk, memory, and VectorDB."""
    # 1. Save Image to Disk
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    person_dir = os.path.join(base_dir, person_name.replace(" ", "_"))
    
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        
    filename = f"{person_name.replace(' ', '_')}_{int(time.time())}.jpg"
    save_path = os.path.join(person_dir, filename)
    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    
    # 2. Update In-Memory Data
    paths.append(save_path)
    
    if cluster_id in clusters:
        clusters[cluster_id].append(save_path)
    else:
        clusters[cluster_id] = [save_path]
        
    # 3. Update HNSW Index
    vector = np.array([embedding]).astype('float32')
    faiss.normalize_L2(vector)
    index.add(vector)
    
    # 4. Save
    save_database(clusters, paths, index)
    return save_path

# --- 3. LOAD STATE ---
# Always reload from disk so changes from other sessions are reflected
# on each Streamlit re-run — avoids stale in-memory data.
try:
    clusters, image_paths, index = load_data()

    # ── Fix 3: Integrity check ──────────────────────────────────────────
    # HNSW indices don't support deletion. If a crash interrupted a write,
    # len(paths) and index.ntotal can go out of sync.  Warn the user early.
    if index is not None and len(image_paths) != index.ntotal:
        st.warning(
            f"⚠️ Data integrity mismatch: {len(image_paths)} paths vs "
            f"{index.ntotal} vectors in the FAISS index. "
            "Results may be unreliable — consider rebuilding the index."
        )

    st.session_state['clusters'] = clusters
    st.session_state['paths'] = image_paths
    st.session_state['index'] = index
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# --- 4. TABS INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🔍 Database Explorer", "➕ Add / Search Face", "🔄 Federated Sync"])

# === TAB 1: EXPLORER (With Search) ===
with tab1:
    st.subheader("View Discovered Identities")
    
    curr_clusters = st.session_state['clusters']
    
    # 1. Build the list of all Names and IDs first
    cluster_names = {}
    valid_ids = []
    
    for c_id, imgs in curr_clusters.items():
        if c_id == -1: continue # Skip noise
        if len(imgs) > 0:
            path = imgs[0]
            name = os.path.basename(os.path.dirname(path)).replace("_", " ")
            cluster_names[c_id] = name
            valid_ids.append(c_id)
            
    valid_ids.sort()

    if not valid_ids:
        st.info("Database is empty.")
    else:
        # 2. Search Box Logic
        col_search, col_space = st.columns([1, 2])
        with col_search:
            search_query = st.text_input("🔍 Search by Name or ID", placeholder="Type 'Ratan' or '12'...")

        # 3. Filter the ID list based on Search
        display_ids = []
        if search_query:
            search_lower = search_query.lower()
            for c_id in valid_ids:
                name = cluster_names[c_id].lower()
                # Check if name matches OR if ID matches
                if search_lower in name or search_query == str(c_id):
                    display_ids.append(c_id)
        else:
            display_ids = valid_ids # Show everyone if search is empty

        # 4. Display Results
        if not display_ids:
            st.warning("No matches found.")
        else:
            if search_query:
                st.caption(f"Found {len(display_ids)} matches.")
                
            sel_id = st.selectbox(
                "Select Person", 
                display_ids, 
                format_func=lambda x: f"{cluster_names[x]} (ID: {x})"
            )
            
            if sel_id is not None:
                imgs = curr_clusters[sel_id]
                st.write(f"📂 **{cluster_names[sel_id]}** - {len(imgs)} images")
                
                cols = st.columns(THUMBNAIL_COLUMNS)
                for i, p in enumerate(imgs):
                    if os.path.exists(p):
                        with cols[i % THUMBNAIL_COLUMNS]:
                            st.image(p, use_container_width=True)

# === TAB 2: SMART ADD / SEARCH ===
with tab2:
    st.subheader("Process New Image")
    
    uploaded_file = st.file_uploader("Upload a face", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        col_preview, col_details = st.columns([1, 2])
        with col_preview:
            st.image(img_rgb, caption="Uploaded Image", width=250)
            
        with col_details:
            if st.button("🔍 Analyze Face"):
                with st.spinner("Analyzing..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                            tf.write(uploaded_file.getvalue())
                            temp_path = tf.name
                            
                        emb_objs = DeepFace.represent(temp_path, model_name='ArcFace', enforce_detection=True)
                        embedding = emb_objs[0]['embedding']
                        os.remove(temp_path)
                        
                        query = np.array([embedding]).astype('float32')
                        faiss.normalize_L2(query)
                        D, I = st.session_state['index'].search(query, k=1)
                        
                        st.session_state['current_embedding'] = embedding
                        st.session_state['current_img'] = img_rgb
                        st.session_state['search_score'] = D[0][0]
                        st.session_state['search_idx'] = I[0][0]
                        
                    except Exception as e:
                        st.error("No face detected! Try a clearer photo.")

        if 'current_embedding' in st.session_state:
            score = st.session_state['search_score']
            idx = st.session_state['search_idx']
            paths_list = st.session_state['paths']
            
            st.divider()
            
            # Identify closest match
            existing_name = "Unknown"
            matched_cluster_id = None
            match_path = None           # ← Fix 2: initialise before conditional

            if idx < len(paths_list):
                match_path = paths_list[idx]
                existing_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
                for c_id, p_list in st.session_state['clusters'].items():
                    if match_path in p_list:
                        matched_cluster_id = c_id
                        break

            # OPTION 1
            st.subheader("Option 1: Add to Existing Person")
            if score > MATCH_THRESHOLD:
                st.success(f"Closest Match: **{existing_name}** (Similarity: {score:.2f})")
            else:
                st.warning(f"Closest Match: **{existing_name}** (Low Similarity: {score:.2f})")

            col_opt1_a, col_opt1_b = st.columns([1, 4])
            with col_opt1_a:
                if match_path and os.path.exists(match_path):   # ← Fix 2: safe guard
                     st.image(match_path, caption="Database Match", width=100)
            with col_opt1_b:
                st.write(f"Do you want to add this to **{existing_name}**?")
                if st.button(f"➕ Add to '{existing_name}'"):
                    # ── Fix 5: dedup check ───────────────────────────────────
                    if score > DEDUP_THRESHOLD:
                        st.warning(
                            f"⚠️ This face is very similar to an existing entry "
                            f"(score {score:.2f} > dedup threshold {DEDUP_THRESHOLD}). "
                            "It may already be in the database."
                        )
                    add_new_face(
                        st.session_state['current_img'],
                        st.session_state['current_embedding'],
                        existing_name,
                        matched_cluster_id,
                        st.session_state['clusters'],
                        st.session_state['paths'],
                        st.session_state['index']
                    )
                    st.success(f"Added to {existing_name}!")
                    del st.session_state['current_embedding']
                    time.sleep(1.5)
                    st.rerun()

            st.divider()

            # OPTION 2
            st.subheader("Option 2: Create New Person")
            st.info("If this is a new person, enter their name below.")
            
            col_opt2_a, col_opt2_b = st.columns([3, 1])
            with col_opt2_a:
                new_name_input = st.text_input("Enter Name:", key="new_name_input")
            with col_opt2_b:
                st.write("##")
                if st.button("✨ Create New Cluster"):
                    if new_name_input.strip():
                        if not st.session_state['clusters']:
                            new_id = 0
                        else:
                            new_id = max(st.session_state['clusters'].keys()) + 1
                        
                        add_new_face(
                            st.session_state['current_img'],
                            st.session_state['current_embedding'],
                            new_name_input,
                            new_id,
                            st.session_state['clusters'],
                            st.session_state['paths'],
                            st.session_state['index']
                        )
                        st.success(f"Created {new_name_input}!")
                        del st.session_state['current_embedding']
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Enter a name.")

# === TAB 3: FEDERATED SYNC ===
with tab3:
    st.subheader("Privacy-Preserving Cluster Sync")
    st.caption(
        "Exchange encrypted identity centroids with another device — "
        "no images ever leave your machine."
    )

    # ── Fix 1: Visible privacy disclaimer ─────────────────────────────────
    with st.expander("⚠️ Privacy & Security Notice", expanded=False):
        st.markdown(
            "**Semi-honest relay model.** In this demo the encryption key is "
            "sent alongside the ciphertext so that the relay server can decrypt "
            "centroids for cosine comparison. The server is trusted *not to leak* "
            "them, but it **does** see plaintext centroid vectors.\n\n"
            "In a production deployment this comparison would happen via "
            "**homomorphic encryption** (e.g. CKKS / TenSEAL) or **secure "
            "multi-party computation** so the relay never sees plaintext.\n\n"
            "Additionally, **differential privacy noise** (ε-DP) is applied to "
            "every centroid before export, making individual-face recovery "
            "statistically impractical even if an adversary intercepts the data."
        )

    SYNC_URL = f"http://{SYNC_SERVER_HOST}:{SYNC_SERVER_PORT}"
    DEVICE_ID = st.text_input(
        "Device ID", value=f"device_{os.getpid()}",
        help="Unique identifier for this device. Use different IDs for the two-device demo."
    )

    curr = st.session_state['clusters']
    curr_paths = st.session_state['paths']
    curr_index = st.session_state['index']

    # ── Step 1: Centroid summary (cached to avoid FAISS reconstruct on every rerun)
    st.markdown("---")
    st.markdown("#### 1 · Centroid Summary")

    if curr_index is not None and curr:
        # Cache centroids in session_state; recompute only when cluster count changes
        cache_key = f"_cached_centroids_{len(curr)}_{sum(len(v) for v in curr.values())}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = compute_centroids(curr, curr_paths, curr_index)
        centroids = st.session_state[cache_key]
        cluster_sizes = {k: len(v) for k, v in curr.items() if k != -1}

        summary_data = []
        for c_id in sorted(centroids.keys()):
            label = os.path.basename(os.path.dirname(curr[c_id][0])).replace("_", " ") if curr.get(c_id) else "?"
            summary_data.append({
                "Cluster ID": c_id,
                "Label": label,
                "Images": cluster_sizes.get(c_id, 0),
                "Centroid Norm": f"{np.linalg.norm(centroids[c_id]):.4f}",
            })
        st.dataframe(summary_data, use_container_width=True)
        st.caption(f"{len(centroids)} clusters ready for sync.")
    else:
        st.info("Database is empty — nothing to sync.")
        centroids = {}
        cluster_sizes = {}

    # ── Step 2: Export & Submit manifest ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 2 · Export & Submit")

    col_eps, col_btn = st.columns([1, 2])
    with col_eps:
        epsilon = st.number_input(
            "Privacy budget (ε)", min_value=0.01, max_value=50.0,
            value=DP_EPSILON, step=0.1,
            help="Lower = more private but noisier centroids."
        )

    with col_btn:
        st.write("##")
        if st.button("🚀 Export & Submit to Relay Server", disabled=not centroids):
            with st.spinner("Generating manifest …"):
                manifest = generate_sync_manifest(
                    device_id=DEVICE_ID,
                    centroids=centroids,
                    cluster_sizes=cluster_sizes,
                    epsilon=epsilon,
                )
                st.session_state['last_manifest'] = manifest

            # Submit to relay
            try:
                resp = requests.post(
                    f"{SYNC_URL}/submit_manifest",
                    data=manifest_to_json(manifest),
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                )
                resp_data = resp.json()
                if resp.status_code == 200:
                    st.success(
                        f"✅ Submitted {resp_data.get('clusters_received', '?')} centroids. "
                        f"Devices registered: {resp_data.get('devices_registered', '?')}"
                    )
                else:
                    st.error(f"Server error: {resp_data.get('message', resp.text)}")
            except requests.ConnectionError:
                st.error(
                    f"Cannot reach sync server at {SYNC_URL}. "
                    "Start it with: `python sync_server.py`"
                )

    # ── Step 3: Merge suggestions ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 3 · Merge Suggestions")

    col_compare, col_fetch = st.columns(2)
    with col_compare:
        if st.button("🔀 Run Cross-Device Comparison"):
            try:
                resp = requests.get(f"{SYNC_URL}/compare", timeout=5)
                data = resp.json()
                st.info(f"Comparison complete — {data.get('total_matches', 0)} matches found.")
            except requests.ConnectionError:
                st.error("Sync server not reachable.")

    with col_fetch:
        if st.button("📥 Fetch My Suggestions"):
            try:
                resp = requests.get(
                    f"{SYNC_URL}/merge_suggestions/{DEVICE_ID}", timeout=5
                )
                suggestions = resp.json().get("suggestions", [])
                st.session_state['sync_suggestions'] = suggestions
                if suggestions:
                    st.success(f"Received {len(suggestions)} merge suggestions.")
                else:
                    st.info("No suggestions yet — ensure both devices have submitted.")
            except requests.ConnectionError:
                st.error("Sync server not reachable.")

    if 'sync_suggestions' in st.session_state and st.session_state['sync_suggestions']:
        previews = preview_merges(
            st.session_state['sync_suggestions'], curr, DEVICE_ID
        )
        st.markdown("**Proposed merges:**")
        st.dataframe(previews, use_container_width=True)

        if st.button("✅ Apply Merges"):
            updated_clusters, updated_paths, count = apply_merge_suggestions(
                st.session_state['sync_suggestions'],
                curr, curr_paths, DEVICE_ID,
            )
            save_merged_state(updated_clusters, updated_paths)
            st.success(f"Applied {count} merges. Reloading…")
            del st.session_state['sync_suggestions']
            time.sleep(1)
            st.rerun()