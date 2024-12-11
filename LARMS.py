import streamlit as st
import torch
import pandas as pd
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddigns_path = "corpus/embeddings.pt"
merged_path = 'corpus/merged_dataset.csv'

def load_or_compute_embeddings(df, model):
    embeddings_file = embeddigns_path
    
    if os.path.exists(embeddings_file):
        context_embeddings = torch.load(embeddings_file, weights_only=True)
        print("Loaded pre-computed embeddings")
    else:
        print("Computing embeddings...")
        contexts = df['Context'].tolist()
        context_embeddings = model.encode(contexts, convert_to_tensor=True)
        torch.save(context_embeddings, embeddings_file)
        print("Saved embeddings to file")
    
    return context_embeddings

# Initialize session states
if 'experiment_mode' not in st.session_state:
    st.session_state.experiment_mode = False
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.title("Large Language Models for Remedying Mental Status")

with st.sidebar:
    st.header("Model Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher values make the output more random, lower values make it more focused and deterministic."
    )
    st.session_state.temperature = temperature
    
    # Add experiment mode toggle to sidebar
    experiment_mode = st.checkbox("Experiment Mode", value=st.session_state.experiment_mode)
    st.session_state.experiment_mode = experiment_mode
    
    # Display current settings
    st.write("Current Settings:")
    st.write(f"- Temperature: {st.session_state.temperature:.1f}")
    st.write(f"- Experiment Mode: {'On' if st.session_state.experiment_mode else 'Off'}")

with st.spinner("Loading dataset..."):
    df = pd.read_csv(merged_path, low_memory=False)

contexts = df['Context'].tolist()
responses = df['Response'].tolist()
context_embeddings = load_or_compute_embeddings(df, embedding_model)

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
        temperature=st.session_state.temperature  # Use the temperature from the slider
    )

# Chat interface
st.subheader("Chat with LARMS")

def chat_input_area():
    user_question = st.text_input("Type your message here...", key="user_input", label_visibility="collapsed")
    return user_question

if st.session_state.conversation_history:
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.chat_message("user").markdown(message['content'])
        else:
            st.chat_message("assistant").markdown(message['content'])

user_question = st.chat_input("Type your message here...")

if user_question:
    # Add user input to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_question})

    # Find the most similar context
    with st.spinner("Finding the most similar context..."):
        similar_context, similar_response, similarity_score = find_most_similar_context(user_question, context_embeddings)
    
    # Show experiment data if enabled
    if st.session_state.experiment_mode:
        with st.spinner("Loading experiment data..."):
            st.write("Similar Context:", similar_context)
            st.write("Suggested Response:", similar_response)
            st.write("Similarity Score:", f"{similarity_score:.4f}")
            st.write("Current Temperature:", f"{st.session_state.temperature:.1f}")
    
    # Construct the prompt
    prompt = f"""You are an AI Powered Chatbot who provide remedies to queries, your remedies should always be confident and never sound lacking. Always sound 
    emotionally strong and give confidence
    to the person that the remedy you provide definitely works. 
    You should not respond to any other kind of questions which are unrelated to mental health and life.

    User question: {user_question}
    Similar context from database: {similar_context}
    Suggested response: {similar_response}
    Similarity score: {similarity_score}
    
    """

    # Generate the AI response
    with st.spinner("Generating AI response..."):
        try:
            response = groq_chat.invoke(st.session_state.conversation_history + [{"role": "user", "content": prompt}])
            ai_response = response.content

            # Add AI response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

            # Display AI response dynamically
            st.chat_message("assistant").markdown(ai_response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
