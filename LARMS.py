import streamlit as st
import torch
import pandas as pd
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')

st.set_page_config(layout="wide")

# Hide Streamlit's default menu bar
st.markdown("""
    <style>
    .css-1v0mbdj.e16nr0p30 {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# File paths
embeddings_path = "corpus/embeddings.pt"
merged_path = 'corpus/merged_dataset.csv'

# Function to load or compute embeddings
def load_or_compute_embeddings(df, model):
    embeddings_file = embeddings_path

    if os.path.exists(embeddings_file):
        context_embeddings = torch.load(embeddings_file)
        print("Loaded pre-computed embeddings")
    else:
        print("Computing embeddings...")
        contexts = df['Context'].tolist()
        context_embeddings = model.encode(contexts, convert_to_tensor=True)
        torch.save(context_embeddings, embeddings_file)
        print("Saved embeddings to file")

    return context_embeddings

st.title("Large Language Models for Remedying Mental Status")

# Load data and embeddings
df = pd.read_csv(merged_path, low_memory=False)

contexts = df['Context'].tolist()
responses = df['Response'].tolist()
context_embeddings = load_or_compute_embeddings(df, embedding_model)

# Function to find the most similar context
def find_most_similar_context(question, context_embeddings):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return contexts[most_similar_idx], responses[most_similar_idx], similarities[0][most_similar_idx].item()

groq_api_key = "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"

if groq_api_key:
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.2-90b-vision-preview",
        temperature=0.7  # Fixed temperature
    )

# Function to compute distinct n-grams
def distinct_ngrams(text, n):
    words = nltk.word_tokenize(text.lower())
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0

# User input and processing
user_question = st.text_input("Type your message here...", key="user_input")

if user_question:
    # Find the most similar context
    with st.spinner("Finding the most similar context..."):
        similar_context, similar_response, similarity_score = find_most_similar_context(user_question, context_embeddings)

    # Construct the prompt
    prompt = f"""You are an AI Powered Chatbot who provides remedies to queries. Your remedies should always be confident and never sound lacking. Always sound \
    emotionally strong and give confidence to the person that the remedy you provide definitely works. \
    You should not respond to any other kind of questions which are unrelated to mental health and life.

    User question: {user_question}
    Similar context from database: {similar_context}
    Suggested response: {similar_response}
    Similarity score: {similarity_score}
    """

    # Generate the AI response
    with st.spinner("Generating AI response..."):
        try:
            response = groq_chat.invoke([{"role": "user", "content": prompt}])
            ai_response = response.content

            # Calculate metrics
            bleu_score = sentence_bleu([similar_response.split()], ai_response.split())
            meteor = meteor_score([nltk.word_tokenize(similar_response)], nltk.word_tokenize(ai_response))

            # N-gram diversity
            distinct_2 = distinct_ngrams(ai_response, 2)
            distinct_3 = distinct_ngrams(ai_response, 3)

            # Display response and metrics
            st.markdown(f"**AI Response:** {ai_response}")
            st.write(f"BLEU Score: {bleu_score:.4f}")
            st.write(f"METEOR Score: {meteor:.4f}")
            st.write(f"Similarity Score: {similarity_score:.4f}")
            st.write(f"Distinct-2: {distinct_2:.4f}")
            st.write(f"Distinct-3: {distinct_3:.4f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
