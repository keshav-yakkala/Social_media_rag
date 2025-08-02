import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os # For managing file paths and potentially environment variables
import uuid # To generate unique IDs for ChromaDB entries
# Define the path to your dataset
DATA_PATH = os.path.join('data', 'Tweets.csv')

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully from {DATA_PATH}. Shape: {df.shape}")
    # Display the first few rows and columns to verify
    print(df.head())
    print(df.columns)
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Make sure the file is in the 'data' folder.")
    exit() # Exit if data not found, can't proceed

# Identify the column containing the tweet text
TEXT_COLUMN = 'text' # Confirmed for this specific dataset

# Check if the text column exists
if TEXT_COLUMN not in df.columns:
    print(f"Error: Column '{TEXT_COLUMN}' not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Drop rows where the text column is missing
df.dropna(subset=[TEXT_COLUMN], inplace=True)
print(f"Shape after dropping NA texts: {df.shape}")

# Convert text column to string type to avoid embedding errors
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

# Optional: Further basic text cleaning (you can expand this later if needed)
# df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: x.lower()) # Example: lowercase
# df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace(r'http\S+', '', regex=True) # Example: remove URLs
# df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace(r'@\w+', '', regex=True) # Example: remove mentions

# Initialize the Sentence Transformer model for embeddings
# 'all-MiniLM-L6-v2' is a good balance of size, speed, and performance.
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")
# Initialize ChromaDB client (persistent client to store data on disk)
# The 'chroma_db' folder will be created in your project directory
CHROMA_DB_PATH = "chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Get or create a collection. A collection is like a table in a traditional DB.
COLLECTION_NAME = "social_media_posts"
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")
# Check if collection is already populated
if collection.count() == 0:
    print("ChromaDB collection is empty. Generating embeddings and populating...")
    # We'll process in batches to save memory and be more efficient
    batch_size = 1000 # Adjust based on your RAM and dataset size
    total_tweets = len(df)
    ids = []
    documents = []
    metadatas = []

    for i in range(0, total_tweets, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        batch_texts = batch_df[TEXT_COLUMN].tolist()

        # Generate embeddings for the batch
        batch_embeddings = model.encode(batch_texts).tolist() # Convert to list for ChromaDB

        # Prepare data for ChromaDB
        for j, text in enumerate(batch_texts):
            original_index = batch_df.index[j]
            ids.append(str(uuid.uuid4())) # Generate a unique ID for each document
            documents.append(text)
            # You can store more metadata here if your dataset has it (e.g., 'sentiment', 'date')
            metadatas.append({
                "original_index": int(original_index),
                "source": "Tweets.csv"
                # Add other relevant columns from df as metadata, e.g.,
                # "airline_sentiment": batch_df.loc[original_index, 'airline_sentiment']
            })

        # Add to ChromaDB
        collection.add(
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Processed {min(i + batch_size, total_tweets)} / {total_tweets} tweets.")

        # Clear lists for next batch
        ids = []
        documents = []
        metadatas = []

    print(f"ChromaDB population complete. Total documents: {collection.count()}")
else:
    print(f"ChromaDB collection '{COLLECTION_NAME}' already contains {collection.count()} documents. Skipping embedding generation.")
    print("You can delete the 'chroma_db' folder and re-run if you want to re-ingest.")

# --- Test a simple retrieval ---
print("\n--- Testing Retrieval ---")
query_text = "bad flight experience"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)
print(f"Query: '{query_text}'")
if results and results['documents']:
    for i, doc in enumerate(results['documents'][0]):
        print(f"  Result {i+1}: {doc}")
else:
    print("No results found for the query.")

