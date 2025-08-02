# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os
from collections import Counter # For trend analysis
import re # For cleaning text for trend analysis

# --- Configuration ---
# Define paths (ensure these match your project structure)
DATA_PATH = os.path.join('data', 'Tweets.csv')
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "social_media_posts"
TEXT_COLUMN = 'text' # Name of the text column in your CSV

# --- Initialize ChromaDB and Embedding Model (only once) ---
@st.cache_resource # Caches the model and client so it's not reloaded on every Streamlit rerun
def initialize_rag_components():
    print("Initializing RAG components (model, chromadb client, collection)...")
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"RAG components initialized. ChromaDB collection '{COLLECTION_NAME}' has {collection.count()} documents.")
    return model, collection

model, collection = initialize_rag_components()

# --- RAG Function ---
def get_rag_response(query_text, num_results=5):
    # 1. Query Embedding
    query_embedding = model.encode(query_text).tolist()

    # 2. Retrieval from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
        include=['documents', 'metadatas']
    )

    retrieved_docs = results['documents'][0] if results and results['documents'] else []
    retrieved_metadatas = results['metadatas'][0] if results and results['metadatas'] else []

    # 3. Context Construction
    context = ""
    if retrieved_docs:
        context += "Here are some relevant social media posts:\n\n"
        for i, doc in enumerate(retrieved_docs):
            context += f"Post {i+1}: {doc}\n"
        context += "\n"
    else:
        context = "No highly relevant posts found for your query. Consider rephrasing."

    # --- IMPORTANT: LLM GENERATION IS BYPASSED DUE TO TECHNICAL/HARDWARE CONSTRAINTS ---
    # The app will NOT attempt to call an external or local LLM with this code.
    # This function will directly return the retrieved context as the response.
    return context, retrieved_docs
    # ************************************************************

# --- Trend Analysis Function ---
@st.cache_data # Caches the dataframe loading so it's fast on reruns
def load_full_data_for_trends():
    try:
        df_full = pd.read_csv(DATA_PATH)
        df_full.dropna(subset=[TEXT_COLUMN], inplace=True)
        df_full[TEXT_COLUMN] = df_full[TEXT_COLUMN].astype(str)
        # Ensure 'tweet_created' is datetime for time-based trends
        df_full['tweet_created'] = pd.to_datetime(df_full['tweet_created'], errors='coerce')
        df_full.dropna(subset=['tweet_created'], inplace=True) # Drop rows where date conversion failed
        return df_full
    except Exception as e:
        st.error(f"Error loading data for trend analysis: {e}")
        return pd.DataFrame()

full_df = load_full_data_for_trends()

def get_trending_topics(df, top_n=10, time_window_hours=24):
    if df.empty:
        return []

    latest_timestamp = df['tweet_created'].max()
    recent_tweets_df = df[df['tweet_created'] >= (latest_timestamp - pd.Timedelta(hours=time_window_hours))]

    if recent_tweets_df.empty:
        st.warning(f"No tweets in the last {trend_time_window} hours relative to latest. Using all {len(df)} tweets for trend analysis.")
        tweets_for_trends = df[TEXT_COLUMN].tolist()
    else:
        tweets_for_trends = recent_tweets_df[TEXT_COLUMN].tolist()

    all_words = []
    all_hashtags = []
    for tweet in tweets_for_trends:
        hashtags = re.findall(r'#(\w+)', tweet)
        all_hashtags.extend(hashtags)

        words = re.findall(r'\b[a-zA-Z]{2,}\b', tweet.lower())
        stop_words = set(
            ["the", "and", "is", "in", "it", "to", "of", "for", "a", "i", "you", "my", "on", "with", "at", "we", "from", "that", "this", "but", "not", "so", "be", "have", "do", "would", "can", "they", "just", "flight", "flights", "airline", "usairways", "united", "americanair", "southwestair", "delta", "jetblue"]
        )
        filtered_words = [word for word in words if word not in stop_words]
        all_words.extend(filtered_words)

    word_counts = Counter(all_words)
    hashtag_counts = Counter(all_hashtags)

    trending = []
    for hashtag, count in hashtag_counts.most_common(top_n):
        trending.append(f"#{hashtag} ({count} mentions)")

    word_idx = 0
    while len(trending) < top_n and word_idx < len(word_counts.most_common()):
        word, count = word_counts.most_common()[word_idx]
        if f"#{word}" not in [t.split(' ')[0] for t in trending]:
            trending.append(f"{word} ({count} mentions)")
        word_idx += 1

    return trending


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Social Media Trend & RAG System")

st.title("✈️ Social Media Trend & RAG System (Airline Tweets)")
st.markdown("Developed by YAKKALA KESAVA SAI AYYAPPA for Nervesparks India Pvt Ltd.")
st.markdown("---")


# --- RAG Section ---
st.header("🔍 Ask the RAG System")
st.write("Enter a query to find relevant tweets and get contextual information.")

user_query = st.text_input("Your Query:", "What are people saying about bad service?")
if st.button("Get Response"):
    if user_query:
        with st.spinner("Searching for relevant posts..."): # Changed text as no LLM generation
            response_text, retrieved_tweets = get_rag_response(user_query)
            st.subheader("Retrieved Context (Relevant Tweets):") # Changed subheader
            st.info(response_text) # Displays the formatted context

            if retrieved_tweets:
                st.subheader("Top Relevant Tweets Found:") # This will be redundant, but keep for clarity. st.info above is the main output.
                for tweet in retrieved_tweets:
                    st.write(f"- {tweet}")
    else:
        st.warning("Please enter a query.")

st.markdown("---")

# --- Trend Analysis Section ---
st.header("📈 Trending Topics")
st.write("Discover what's currently trending in the social media feed.")

trend_time_window = st.slider(
    "Simulate 'Recent' Trends (hours within dataset's latest tweet):",
    min_value=1, max_value=720, value=24, step=12,
    help="Adjust this to see trends over different simulated recent periods in the dataset."
)

if st.button("Identify Trends"):
    if not full_df.empty:
        with st.spinner(f"Analyzing trends in the last {trend_time_window} hours of data..."):
            trending_topics = get_trending_topics(full_df, top_n=10, time_window_hours=trend_time_window)
            if trending_topics:
                st.subheader(f"Top {len(trending_topics)} Trending Topics:")
                for i, topic in enumerate(trending_topics):
                    st.write(f"{i+1}. {topic}")
            else:
                st.info("No distinct trends identified for the selected period.")
    else:
        st.error("Cannot analyze trends: Data not loaded or is empty.")

st.markdown("---")
st.caption("Note: For a full RAG system, an LLM would synthesize the retrieved information into more natural language responses, but this feature is currently bypassed due to technical limitations.") # Updated caption