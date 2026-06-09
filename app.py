import os
import pandas as pd
import numpy as np
import urllib.request
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# App setup
st.set_page_config(page_title="Product Search Engine", page_icon="🛍️", layout="wide")
st.title("🛍️ Multimodal Semantic Product Search Engine")

# Load CLIP model
@st.cache_resource
def get_model():
    return SentenceTransformer('clip-ViT-B-32')

# Load and clean dataset
@st.cache_data
def get_data():
    csv_file = "fashion.csv"
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Automated fallback if local file is missing
        backup = {
            'ProductId': [1001, 1002, 1003, 1004, 1005],
            'ProductType': ['Running Shoes', 'Sneakers', 'Canvas', 'Boots', 'Loafers'],
            'Gender': ['Men', 'Men', 'Women', 'Unisex', 'Women'],
            'ProductTitle': ['Asics Black Gel', 'Nike Red Air Force', 'Classic White Vans', 'Timberland Waterproof', 'Puma Navy Blue Mesh']
        }
        df = pd.DataFrame(backup)
        df.to_csv(csv_file, index=False)
        return df

    df = df[['ProductId', 'ProductType', 'Gender', 'ProductTitle']].dropna().head(300)
    df['text_profile'] = df['Gender'] + " " + df['ProductType'] + " " + df['ProductTitle']
    return df

model = get_model()
df = get_data()

# Precompute embeddings
@st.cache_data
def get_embeddings(_model, _df):
    texts = _df['text_profile'].tolist()
    return _model.encode(texts, batch_size=32, show_progress_bar=False)

catalog_features = get_embeddings(model, df)

# Search Function
def search_catalog(query_vec, features, original_df, k=6):
    sim_scores = cosine_similarity(query_vec.reshape(1, -1), features).flatten()
    top_k_idx = np.argsort(sim_scores)[::-1][:k]
    
    results = original_df.iloc[top_k_idx].copy()
    results['score'] = sim_scores[top_k_idx]
    return results

# Sidebar UI
st.sidebar.header("Search Panel")
mode = st.sidebar.radio("Input Type:", ("Text", "Image"))

query_vec = None

if mode == "Text":
    user_query = st.sidebar.text_input("Search here:", placeholder="Type product description...")
    if user_query.strip():
        query_vec = model.encode(user_query)
else:
    uploaded_img = st.sidebar.file_uploader("Upload product image:", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img)
        st.sidebar.image(img, caption="Query Image", use_container_width=True)
        query_vec = model.encode(img)

# Main Dashboard
if query_vec is not None:
    st.subheader("Search Results")
    matches = search_catalog(query_vec, catalog_features, df, k=6)
    
    cols = st.columns(3)
    for idx, item in enumerate(matches.itertuples()):
        with cols[idx % 3]:
            st.markdown(f"### Item #{item.ProductId}")
            st.write(f"**Category:** {item.ProductType} ({item.Gender})")
            st.write(f"**Title:** {item.ProductTitle}")
            
            score_val = max(0.0, min(1.0, float(item.score)))
            st.progress(score_val)
            st.write(f"Score: **{score_val * 100:.2f}%**")
            st.markdown("---")
else:
    st.info("Please use the sidebar to enter a text query or upload an image to start searching.")