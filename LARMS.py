import os
import streamlit as st
import torch
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
import textstat
from textblob import TextBlob
import dotenv

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

# Configure Streamlit page
st.set_page_config(layout="wide", page_title="Mental Health Support Chatbot")

# Hide Streamlit's default menu bar
st.markdown("""
    <style>
    .css-1v0mbdj.e16nr0p30 {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
dotenv.load_dotenv()

# Configuration
BASE_DIR = os.path.expanduser("/")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "corpus/embeddings.pt")
MERGED_PATH = os.path.join(BASE_DIR, "corpus/merged_dataset.csv")

# Ensure paths exist
def ensure_path_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

ensure_path_exists(EMBEDDINGS_PATH)
ensure_path_exists(MERGED_PATH)

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Groq Integration
def initialize_groq(api_key, model_name, temperature):
    try:
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"Groq initialization error: {e}")
        return None

# Function to load or compute embeddings
def load_or_compute_embeddings(df, model):
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            return torch.load(EMBEDDINGS_PATH)
        else:
            embeddings = model.encode(df['Context'].tolist(), convert_to_tensor=True)
            torch.save(embeddings, EMBEDDINGS_PATH)
            return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {e}")
        return None

# Find the most similar context
def find_most_similar_context(user_input, context_embeddings):
    try:
        user_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, context_embeddings)[0]
        max_idx = torch.argmax(similarities).item()
        return contexts[max_idx], responses[max_idx], similarities[max_idx].item()
    except Exception as e:
        st.error(f"Error finding similar context: {e}")
        return None, None, None

# Evaluate the AI response
def evaluate_response(target_response, ai_response):
    try:
        # Calculate semantic similarity
        target_embedding = embedding_model.encode(target_response, convert_to_tensor=True)
        ai_embedding = embedding_model.encode(ai_response, convert_to_tensor=True)
        semantic_similarity = float(util.pytorch_cos_sim(target_embedding, ai_embedding)[0][0])

        # Sentiment analysis
        target_sentiment = TextBlob(target_response).sentiment.polarity
        ai_sentiment = TextBlob(ai_response).sentiment.polarity
        sentiment_consistency = 1 - abs(target_sentiment - ai_sentiment)

        # Language diversity
        ai_words = ai_response.split()
        unique_word_ratio = len(set(ai_words)) / len(ai_words)
        distinct_1 = len(set(ai_words)) / len(ai_words)
        distinct_2 = len(set(zip(ai_words, ai_words[1:]))) / len(ai_words)
        distinct_3 = len(set(zip(ai_words, ai_words[1:], ai_words[2:]))) / len(ai_words)

        # Readability metrics
        flesch_reading_ease = textstat.flesch_reading_ease(ai_response)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(ai_response)
        gunning_fog = textstat.gunning_fog(ai_response)

        return {
            "semantic_similarity": semantic_similarity,
            "sentiment_consistency": sentiment_consistency,
            "diversity": {
                "unique_word_ratio": unique_word_ratio,
                "distinct_1": distinct_1,
                "distinct_2": distinct_2,
                "distinct_3": distinct_3,
            },
            "readability": {
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "gunning_fog": gunning_fog,
            },
        }
    except Exception as e:
        st.error(f"Error evaluating response: {e}")
        return {}

# Main Streamlit app
def main():
    st.title("Large Language Models for Remedying Mental Status")

    # Load data and embeddings
    try:
        df = pd.read_csv(MERGED_PATH, low_memory=False)
        global contexts, responses
        contexts = df['Context'].tolist()
        responses = df['Response'].tolist()
        context_embeddings = load_or_compute_embeddings(df, embedding_model)
        if context_embeddings is None:
            st.error("Failed to load or compute embeddings.")
            return
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return

    # User input
    user_question = st.text_input("Type your message here...", key="user_input")

    # Temperature slider
    temperature = st.slider("Select the temperature for response generation:", 0.1, 1.0, 0.7)

    # Model selection
    model_options = [
        "llama-3.3-70b-specdec",
        "llama-3.3-70b-versatile",
        "distil-whisper-large-v3-en",
        "gemma2-9b-it",
        "llama-3.2-90b-vision-preview"
    ]
    selected_model = st.selectbox("Select the model for response generation:", model_options)

    if user_question:
        # Find the most similar context
        with st.spinner("Finding the most similar context..."):
            similar_context, similar_response, similarity_score = find_most_similar_context(
                user_question, context_embeddings
            )
            if similar_context is None:
                st.error("Could not find a similar context.")
                return

        # Construct the prompt
        prompt = f"""
        You are a LLM purposefully designed only for Mental Health Conversations
        User question: {user_question}
        Context: {similar_context}
        Response: {similar_response}"""

        # Generate the AI response
        with st.spinner("Generating AI response..."):
            try:
                groq_chat = initialize_groq(os.getenv("GROQ_API_KEY", "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"), selected_model, temperature)
                response = groq_chat.invoke([{"role": "user", "content": prompt}])
                ai_response = response.content

                # Evaluate the response
                evaluation_metrics = evaluate_response(similar_response, ai_response)

                # Display response and metrics
                st.markdown(f"**AI Response:** {ai_response}")
                with st.expander("Response Metrics"):
                    st.write(evaluation_metrics)
            except Exception as e:
                st.error(f"Response generation error: {e}")

if __name__ == "__main__":
    main()
