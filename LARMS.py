import streamlit as st
import torch
import pandas as pd
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .css-1v0mbdj.e16nr0p30 {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings_path = "corpus/embeddings.pt"
merged_path = 'corpus/merged_dataset.csv'

def load_or_compute_embeddings(df, model):
    embeddings_file = embeddings_path
    
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
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'chats' not in st.session_state:
    st.session_state.chats = {}

if 'current_chat' not in st.session_state:
    st.session_state.current_chat = "default"

st.title("Large Language Models for Remedying Mental Status")

# Sidebar Chat Management Section
with st.sidebar:
    st.header("Chat Management")
    chat_name = st.text_input("Enter new chat name:", key="new_chat_name")
    
    if st.button("Create New Chat") and chat_name:
        if chat_name not in st.session_state.chats:
            st.session_state.chats[chat_name] = []
            st.session_state.current_chat = chat_name
        else:
            st.warning("Chat name already exists.")

    if st.session_state.chats:
        st.selectbox("Select Chat:", options=list(st.session_state.chats.keys()), key="chat_selector", 
                     on_change=lambda: st.session_state.update({"current_chat": st.session_state.chat_selector}))

    st.markdown("---")
    st.button("Delete Current Chat", 
              on_click=lambda: st.session_state.chats.pop(st.session_state.current_chat, None) and st.session_state.update({"current_chat": "default"}))

# Load data and embeddings
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
        temperature=0.7  # Fixed temperature
    )

# Chat interface
def chat_input_area():
    user_question = st.text_input("Type your message here...", key="user_input", label_visibility="collapsed")
    return user_question

# Update the chat history of the current chat
if st.session_state.current_chat in st.session_state.chats:
    if user_question := st.chat_input("Type your message here..."):
        st.session_state.chats[st.session_state.current_chat].append({"role": "user", "content": user_question})

        # Find the most similar context
        with st.spinner("Finding the most similar context..."):
            similar_context, similar_response, similarity_score = find_most_similar_context(user_question, context_embeddings)

        # Construct the prompt
        prompt = f"""You are an AI Powered Chatbot who provides remedies to queries. Your remedies should always be confident and never sound lacking. Always sound 
        emotionally strong and give confidence to the person that the remedy you provide definitely works. 
        You should not respond to any other kind of questions which are unrelated to mental health and life.
        If Similarity is low ignore the things below (Do not mention this in the response)

        User question: {user_question}
        Similar context from database: {similar_context}
        Suggested response: {similar_response}
        Similarity score: {similarity_score}
        
        """

        # Generate the AI response
        with st.spinner("Generating AI response..."):
            try:
                response = groq_chat.invoke(st.session_state.chats[st.session_state.current_chat] + [{"role": "user", "content": prompt}])
                ai_response = response.content

                # Add AI response to conversation history
                st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": ai_response})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display current chat history dynamically
for message in st.session_state.chats.get(st.session_state.current_chat, []):
    if message['role'] == 'assistant':
        st.chat_message("assistant").markdown(message['content'])
    elif message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
