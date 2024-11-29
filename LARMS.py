import streamlit as st
import torch
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import time

# Hugging Face API Token (Replace with your token)
HF_TOKEN = "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"

# Initialize the embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading model: {e}")

def initialize_inference_client(model_name, token):
    """
    Initialize Hugging Face Inference Client with error handling
    """
    try:
        client = InferenceClient(model=model_name, token=token)
        return client
    except Exception as e:
        st.error(f"Client Initialization Error: {e}")
        return None


@st.cache_data
def load_dataset(file_path):
    """
    Load the dataset from the provided file path.
    """
    return pd.read_csv(file_path, low_memory=False)


def load_or_compute_embeddings(df, model):
    """
    Load or compute embeddings and refresh them based on a 30-day threshold.
    """
    embeddings_file = 'corpus/embeddings.pt'
    
    if os.path.exists(embeddings_file):
        try:
            file_age = time.time() - os.path.getmtime(embeddings_file)
            if file_age > 30 * 24 * 60 * 60:  # Refresh embeddings after 30 days
                os.remove(embeddings_file)
        except Exception as e:
            st.warning(f"Could not check embedding file age: {e}")
    
    if os.path.exists(embeddings_file):
        context_embeddings = torch.load(embeddings_file)
    else:
        contexts = df['Context'].tolist()
        context_embeddings = model.encode(contexts, convert_to_tensor=True)
        torch.save(context_embeddings, embeddings_file)
    
    return context_embeddings


def find_most_similar_context(question, context_embeddings, contexts, responses):
    """
    Find the most similar context in the dataset for the given question.
    """
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return (
        contexts[most_similar_idx],
        responses[most_similar_idx],
        similarities[0][most_similar_idx].item()
    )


def generate_model_response(client, prompt, temperature=0.4, max_tokens=500):
    """
    Generate a response using the Hugging Face Inference Client.
    """
    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        return response
    except Exception as e:
        st.error(f"Response Generation Error: {e}")
        return "I'm experiencing some difficulties right now. Please try again later."


def main():
    st.title("Large Language Models for Remedying Mental Status (LARMS)")
    
    # Sidebar
    with st.sidebar:
        st.header("Model Settings")
        
        # Model selection
        model_name = st.selectbox(
            "Select LLM:",
            options=[
                "mistralai/Mistral-7B-Instruct-v0.2",
                "tiiuae/falcon-7b-instruct",
                "gpt2",
                "EleutherAI/gpt-neo-1.3B",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "meta-llama/Llama-3.2-1B",
                ],
            help="Choose a model to process your query."
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Higher values make the output more random; lower values make it more deterministic."
        )
        
        # Experiment mode toggle
        experiment_mode = st.checkbox("Enable Experiment Mode", value=False)
    
    # Session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Load dataset
    df = load_dataset('corpus/merged_dataset.csv')
    contexts = df['Context'].tolist()
    responses = df['Response'].tolist()
    
    # Compute or load embeddings
    context_embeddings = load_or_compute_embeddings(df, embedding_model)
    
    # Initialize inference client with selected model
    client = initialize_inference_client(model_name, HF_TOKEN)
    
    # User input
    user_question = st.text_area("How are you feeling today?")
    
    if user_question and client:
        # Find most similar context
        with st.spinner("Finding the most similar context..."):
            similar_context, similar_response, similarity_score = find_most_similar_context(
                user_question, context_embeddings, contexts, responses
            )
        
        if experiment_mode:
            st.write("**Similar Context:**", similar_context)
            st.write("**Suggested Response:**", similar_response)
            st.write("**Similarity Score:**", f"{similarity_score:.4f}")
        
        # Construct prompt
        prompt = f""""You are an AI-powered chatbot who provides remedies to queries. "
            "Your remedies should always be confident and emotionally supportive. "
            "Focus on mental health and provide empathetic, actionable advice."
        User Question: {user_question}
        Context: {similar_context}
        Response: {similar_response}
        """
        
        # Generate response
        with st.spinner("Generating AI response..."):
            ai_response = generate_model_response(client, prompt, temperature=temperature)
            st.write("**AI's Response:**", ai_response)
            
            # Update conversation history
            st.session_state.conversation_history.append({
                "user": user_question,
                "ai": ai_response,
                "model": model_name
            })
    
    # Display previous interactions
    if st.session_state.conversation_history:
        with st.expander("Previous Interactions"):
            for idx, interaction in enumerate(st.session_state.conversation_history):
                st.markdown(f"**Interaction {idx + 1}:**")
                st.markdown(f"**Model:** {interaction['model']}")
                st.markdown(f"**User Question:** {interaction['user']}")
                st.markdown(f"**AI Response:** {interaction['ai']}")
                st.markdown("---")


if __name__ == "__main__":
    main()
