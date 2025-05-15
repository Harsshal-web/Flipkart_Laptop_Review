import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

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

# Load or initialize search history
history_file = "search_history.json"
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        search_history = json.load(f)
else:
    search_history = []

# Save search history
def save_search_history(query):
    search_history.append(query)
    if len(search_history) > 10:
        search_history.pop(0)
    with open(history_file, "w") as f:
        json.dump(search_history, f)

# UI
st.title("🔎 Flipkart Laptop Semantic Search Engine")
st.markdown("Type a natural language query to find laptops (e.g., **'best rated HP laptops under 40000'** or **'budget laptop with good battery'**)")

# User input
user_query = st.text_input("💬 Your Query:", "show me best reviewed laptops")

# Display previous search suggestions
if search_history:
    st.subheader("🔄 Try a previous search:")
    for idx, query in enumerate(reversed(search_history), 1):
        st.markdown(f"**{idx}.** {query}")

# Handle query
if user_query:
    save_search_history(user_query)
    
    query_embedding = model.encode([user_query])
    D, I = index.search(query_embedding, k=50)  # Get top 50 for better filtering

    filtered_results = []

    for idx in I[0]:
        result = df.iloc[idx]
        product_name = result['product_name'].lower()
        price = float(result['price_rs'])
        battery = float(result['battery_hrs'])

        # Filtering logic
        query_lower = user_query.lower()

        if "under 40000" in query_lower and price > 40000:
            continue
        if "below 40000" in query_lower and price > 40000:
            continue
        if "above 60000" in query_lower and price < 60000:
            continue
        if "apple" in query_lower and "apple" in product_name and price < 60000:
            continue
        if "good battery" in query_lower and battery < 6:
            continue
        if "battery above 8" in query_lower and battery < 8:
            continue
        if "price below 40000" in query_lower and price >= 40000:
            continue

        filtered_results.append(result)
        if len(filtered_results) >= 10:
            break

    st.subheader("📋 Top Results:")
    if filtered_results:
        for result in filtered_results:
            st.markdown(f"""
            **🖥 Product:** {result['product_name']}  
            **⭐ Overall Rating:** {result['overall_rating']}  
            **💵 Price:** ₹{int(result['price_rs']):,}  
            **🔋 Battery Life:** {result['battery_hrs']} hours  
            **📝 Title:** {result['title']}  
            **💬 Review:** {result['review']}  
            ---  
            """)
    else:
        st.warning("❌ No results match your filters.")
