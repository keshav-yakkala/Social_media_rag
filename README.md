# Social Media RAG with Trend Analysis

## Project Overview
This project develops a Retrieval-Augmented Generation (RAG) system focused on social media content, specifically airline tweets. It identifies trending topics and provides contextual information about viral content, memes, and social movements based on retrieved posts. The application is built using Streamlit for the user interface, Sentence Transformers for embeddings, and ChromaDB for vector storage.

## Key Features Implemented
* **Social Media Content Ingestion:** Processes a dataset of airline tweets (`Tweets.csv`).
* **Trending Topic Identification:** Analyzes tweet content and timestamps to identify popular keywords and hashtags within a configurable time window.
* **Retrieval-Augmented Generation (RAG) - Retrieval Component:** Uses vector embeddings and ChromaDB to find and present highly relevant social media posts in response to user queries. This demonstrates the "retrieval" aspect of RAG effectively.
* **Intuitive User Interface:** A Streamlit web application allows users to interact with the system.

## Technical Stack
* **Python:** Core programming language
* **Streamlit:** For building the interactive web application.
* **Sentence-Transformers:** Used for generating semantic embeddings (`all-MiniLM-L6-v2` model).
* **ChromaDB:** Local vector database for efficient retrieval of relevant tweets.
* **(Attempted) LLMs:** OpenAI, Google Gemini, Ollama (local) - for the generation component of RAG.

## Setup and Local Run Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your_github_username/your_repo_name.git](https://github.com/your_github_username/your_repo_name.git)
    cd your_repo_name
    ```
    (Replace `your_github_username` and `your_repo_name` with your actual GitHub details.)

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows (Command Prompt):
    venv\Scripts\activate
    # On Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    # On Linux/macOS:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Dataset:**
    * Download `Tweets.csv` from [Kaggle: Twitter Data of US Airlines](https://www.kaggle.com/datasets/sriharinagireddy/twitter-data-of-us-airlines).
    * Place the `Tweets.csv` file into a new `data/` folder inside your project directory (e.g., `your_repo_name/data/Tweets.csv`).

5.  **Run Data Ingestion (Build ChromaDB):**
    * This script will generate embeddings and populate the ChromaDB. It might take several minutes.
    ```bash
    python main.py
    ```
    * A `chroma_db/` folder will be created. Do not delete it.

6.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    * The application will open in your web browser (usually at `http://localhost:8501`).

## Summary of Approach

* **Data Ingestion:** Used `Tweets.csv` from Kaggle to simulate social media posts, avoiding complex live API integrations due to time constraints.
* **Embedding Strategy:** Employed `SentenceTransformer('all-MiniLM-L6-v2')` for generating semantic embeddings, striking a balance between performance and accuracy.
* **Vector Database:** Utilized `ChromaDB` for efficient storage and retrieval of tweet embeddings.
* **Chunking Strategy:** Each tweet was treated as an individual chunk, as they are naturally short and self-contained units of information.
* **Trend Analysis:** Implemented a keyword and hashtag frequency analysis over a configurable time window (simulated "recent" period within the historical dataset) to identify trending topics.
* **UX Design:** Designed a clear Streamlit interface for querying the RAG system and viewing trends, focusing on functionality and clarity for the core features.

## Assumptions Made
* The provided `Tweets.csv` dataset adequately represents social media content for this assignment.
* "Real-time" trend monitoring is simulated using a time window within a static dataset for demonstration purposes.
* Basic filtering for trend analysis is sufficient for the scope of this assignment.

## Technical Challenges & Future Enhancements

**Challenge: LLM Integration for Context-Aware Generation**
While the retrieval component of the RAG system is fully functional and demonstrates the core "R" (Retrieval) aspect, I encountered significant and persistent challenges in integrating a live Large Language Model (LLM) for the "G" (Generation) part, specifically synthesizing retrieved information into natural language responses as required by "context-aware generation."

* **Attempt with OpenAI:** Initial attempts with OpenAI's API were halted due to `insufficient_quota` errors, requiring a paid account or renewed free credits.
* **Attempt with Google Gemini (Free Tier):** Switched to Google's Gemini API (via Google AI Studio) for a free alternative. Despite successfully obtaining an API key, placing it securely in `.env`, correctly configuring `google-generativeai`, enabling the "Generative Language API" in the Google Cloud Project, and upgrading the library (`google-generativeai==0.8.5`), the system consistently returned a `404 models/gemini-pro is not found for API version v1beta` error. This specific `404` persisted even after trying model aliases (`gemini-1.0-pro`), strongly suggesting a complex API-side or regional issue beyond simple configuration. **(A direct test in Google AI Studio's web interface confirmed my account *could* access the model, indicating the issue was specific to my local environment's API connection.)**
* **Attempt with Ollama (Local LLM):** As a final attempt to bypass external API issues, I tried running a local LLM using Ollama (specifically the Mistral model). However, this was met with a `500 Internal Server Error: model requires more system memory (4.9 GiB) than is available (2.8 GiB)`, indicating a **hardware limitation** (insufficient RAM on my machine) preventing the model from loading and running locally.

**Impact & Current State:** Due to these persistent, unresolvable external API connectivity issues and a hardware limitation for local LLMs, all attempts to implement the "Generation" component of the RAG system within the strict 3-day deadline were unsuccessful. Consequently, the "Generation" component is currently bypassed in the deployed demo. The system accurately returns the raw retrieved context (relevant tweets) to the user, demonstrating the robust "Retrieval" functionality as a key part of the RAG system. The "Trend Analysis" feature is fully operational as well.

**Future Enhancements:**
* **Resolve LLM Integration:** Diagnose and fix the specific `404` error with Gemini (e.g., through Google Cloud support or further deep-dive network diagnostics), or integrate an alternative LLM (e.g., explore other open-source LLMs if better hardware becomes available or different API providers).
* **Advanced Trend Detection:** Implement more sophisticated trend detection algorithms (e.g., burst detection, NLP-based topic modeling beyond simple keywords).
* **Real-time Data Ingestion:** Integrate with actual social media APIs (Twitter, Reddit, etc.) with robust rate-limit handling.
* **Content Moderation:** Implement a content moderation layer for filtering inappropriate content.
* **Sentiment Analysis:** Integrate deeper sentiment analysis for trending topics and viral data.
* **Multimodal Data:** Expand to analyze images and videos in social media posts.
* **Scalability:** Enhance the system for handling larger volumes of data and higher query loads.

## Evaluation (Basic Metrics)
* **Retrieval Accuracy:** Qualitative assessment confirmed that the ChromaDB returns highly relevant tweets for given queries (e.g., "bad flight experience" consistently returned tweets about negative flight experiences).
* **Latency:** The local application demonstrates low latency for retrieval and trend analysis, with initial model loading and embedding generation being a one-time process.

## Public Link to Working Application
*(This will be added after deployment to HuggingFace Spaces)*

---
*Developed by YAKKALA KESAVA SAI AYYAPPA*