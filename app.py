import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Set page config
st.set_page_config(page_title="Flipkart Laptop Search", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and prepare Data
@st.cache_data
def load_data():
    df = pd.read_csv("Flipkart_Laptop_Review.csv")
    df.fillna("", inplace=True)
    df["combined"] = df["product_name"] + " " + df["title"] + " " + df["review"]
    return df

df = load_data()

# Generate embeddings and create FAISS index
@st.cache_resource
def build_index(documents):
    embeddings = model.encode(documents, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_index(df["combined"].tolist())

# UI
st.title("🔎 Flipkart Laptop Semantic Search Engine")
st.markdown("Type a natural language query to find laptops (e.g., **'best rated HP laptops under 40000'** or **'budget laptop with good battery'**)")

# User input
user_query = st.text_input("💬 Your Query:", "show me best reviewed laptops")

# Handle query
if user_query:
    query_embedding = model.encode([user_query])
    D, I = index.search(query_embedding, k=10)

    st.subheader("📋 Top 10 Results:")
    for idx in I[0]:
        result = df.iloc[idx]
        st.markdown(f"""
        **🖥 Product:** {result['product_name']}  
        **⭐ Overall Rating:** {result['overall_rating']}  
        **📝 Title:** {result['title']}  
        **💬 Review:** {result['review']}  
        ---  
        """)
