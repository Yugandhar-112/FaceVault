import streamlit as st
import pickle
import faiss
import numpy as np
from deepface import DeepFace
import cv2
import os
import tempfile

from config import DATA_FILE, INDEX_FILE, FLAT_INDEX_FILE, IMAGES_DIR, APP_TITLE, PAGE_ICON, THUMBNAIL_COLUMNS

# --- 1. Setup & Configuration ---
st.set_page_config(page_title=f"{APP_TITLE} (Flat Index)", page_icon=PAGE_ICON, layout="wide")

st.title(f"{PAGE_ICON} {APP_TITLE} — Flat FAISS Explorer")
st.caption("Read-only explorer using a flat (non-HNSW) FAISS index. Use `app.py` to add new faces.")

# --- LOAD DATA (With Local Path Fixer) ---
def load_data():
    # 1. Load the raw data
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    # Prefer the dedicated flat index; fall back to the HNSW index file.
    idx_path = FLAT_INDEX_FILE if os.path.exists(FLAT_INDEX_FILE) else INDEX_FILE
    index = faiss.read_index(idx_path)
    
    # 2. FIX PATHS for Local Machine
    # The saved paths point to /content/lfw_extracted/... (Colab).
    # We need them to point to ./images/... (Your Laptop).
    
    old_paths = data['paths']
    new_paths = []
    
    # We assume the 'images' folder is in the same directory as this script
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    
    # -- Fix the List of All Paths --
    for p in old_paths:
        # Extract the folder name (Person) and file name (Image)
        filename = os.path.basename(p) 
        parent_folder = os.path.basename(os.path.dirname(p)) 
        
        # Create new local path
        local_path = os.path.join(base_dir, parent_folder, filename)
        new_paths.append(local_path)
        
    # -- Fix the Clusters Dictionary --
    new_clusters = {}
    for cluster_id, path_list in data['clusters'].items():
        new_list = []
        for p in path_list:
            filename = os.path.basename(p)
            parent_folder = os.path.basename(os.path.dirname(p))
            local_path = os.path.join(base_dir, parent_folder, filename)
            new_list.append(local_path)
        new_clusters[cluster_id] = new_list
        
    return new_clusters, new_paths, index

try:
    clusters, image_paths, index = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure 'app_data.pkl', 'vector_index.bin', and the 'images' folder are in this directory.")
    st.stop()

# --- 2. INTERFACE: TABS ---
tab1, tab2 = st.tabs(["🔍 Cluster Explorer", "🔎 Face Search"])

# --- TAB 1: CLUSTER EXPLORER ---
with tab1:
    st.write("View the unique identities discovered by the AI.")
    
    # Filter out noise (-1)
    valid_cluster_ids = [c for c in clusters.keys() if c != -1]
    valid_cluster_ids.sort()
    
    # --- NEW: Generate Names from Folder Paths ---
    cluster_names = {}
    for c_id in valid_cluster_ids:
        # Get the first image in this cluster to check its folder name
        if len(clusters[c_id]) > 0:
            first_img_path = clusters[c_id][0]
            # Extract the parent folder name (which is the Person's Name)
            person_name = os.path.basename(os.path.dirname(first_img_path))
            # Replace underscores with spaces for better readability
            clean_name = person_name.replace("_", " ")
            cluster_names[c_id] = clean_name
        else:
            cluster_names[c_id] = f"Unknown ID {c_id}"

    # --- NEW: Enhanced Dropdown ---
    # We use 'format_func' to show the Name instead of just the ID number
    selected_cluster = st.selectbox(
        "Select a Person", 
        valid_cluster_ids, 
        format_func=lambda x: f"{cluster_names.get(x, 'Unknown')} (ID: {x})"
    )
    
    if selected_cluster is not None:
        images = clusters[selected_cluster]
        name = cluster_names.get(selected_cluster, "Unknown")
        
        st.info(f"Found {len(images)} photos of **{name}**.")
        
        # Display images in a grid
        cols = st.columns(THUMBNAIL_COLUMNS)
        for i, img_path in enumerate(images):
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    with cols[i % THUMBNAIL_COLUMNS]:
                        st.image(img, use_container_width=True)
            else:
                with cols[i % THUMBNAIL_COLUMNS]:
                    st.error(f"Missing")

# --- TAB 2: FACE SEARCH ---
with tab2:
    st.write("Upload a photo to find who it looks like in the database.")
    
    uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Query Image", width=250)
            
        with col2:
            if st.button("Search Database"):
                with st.spinner('Analyzing face...'):
                    try:
                        # Generate Embedding
                        embedding_objs = DeepFace.represent(
                            img_path=tmp_path,
                            model_name='ArcFace',
                            enforce_detection=True
                        )
                        embedding = embedding_objs[0]['embedding']
                        
                        # Search Index
                        query_vector = np.array([embedding]).astype('float32')
                        faiss.normalize_L2(query_vector)
                        
                        k = 5
                        distances, indices = index.search(query_vector, k)
                        
                        st.success("Search Complete!")
                        st.write("**Top 5 Matches:**")
                        
                        # Show Results
                        result_cols = st.columns(5)
                        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                            if idx < len(image_paths):
                                match_path = image_paths[idx]
                                if os.path.exists(match_path):
                                    match_img = cv2.imread(match_path)
                                    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                                    # Extract name for caption
                                    match_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
                                    
                                    with result_cols[i]:
                                        st.image(match_img, caption=f"{match_name}\n(Score: {dist:.2f})")
                                else:
                                    st.warning("File missing")
                    except Exception as e:
                        st.error(f"Could not process image: {e}")
                        st.info("Try a clearer photo.")
        
        os.remove(tmp_path)