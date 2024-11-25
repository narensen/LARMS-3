import streamlit as st
import torch
import pandas as pd
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util

def load_or_compute_embeddings(df, model):
    embeddings_file = '/home/naren/Documents/LARMS/corpus/embeddings.pt'
    
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

if 'experiment_mode' not in st.session_state:
    st.session_state.experiment_mode = False
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.4


df = pd.read_csv("/home/naren/Documents/LARMS/corpus/merged_dataset.csv")

contexts = df['Context'].tolist()
responses = df['Response'].tolist()

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
    experiment_mode = st.toggle("Experiment Mode", value=st.session_state.experiment_mode)
    st.session_state.experiment_mode = experiment_mode
    
    # Display current settings
    st.write("Current Settings:")
    st.write(f"- Temperature: {st.session_state.temperature:.1f}")
    st.write(f"- Experiment Mode: {'On' if st.session_state.experiment_mode else 'Off'}")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
context_embeddings = load_or_compute_embeddings(df, embedding_model)

def find_most_similar_context(question, context_embeddings):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return contexts[most_similar_idx], responses[most_similar_idx], similarities[0][most_similar_idx].item()

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

groq_api_key = "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"

if groq_api_key:
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.2-90b-vision-preview",
        temperature=st.session_state.temperature
    )

    user_question = st.text_area("How are you feeling today?")

    if user_question:
        similar_context, similar_response, similarity_score = find_most_similar_context(user_question, context_embeddings)
        
        # Show experiment data if experiment mode is enabled
        if st.session_state.experiment_mode:
            with st.expander("Experiment Data", expanded=True):
                st.write("Similar Context:", similar_context)
                st.write("Suggested Response:", similar_response)
                st.write("Similarity Score:", f"{similarity_score:.4f}")
                st.write("Current Temperature:", f"{st.session_state.temperature:.1f}")
        
        prompt = f"""You are an AI-powered chatbot named Lifey or virtual assistant that leverages natural language understanding and empathy to provide mental health and emotional support. You should not respond to any other kind of questions which are unrelated to mental health and life.

        User question: {user_question}
        Similar context from database: {similar_context}
        Suggested response: {similar_response}
        Similarity score: {similarity_score}
        Current temperature: {st.session_state.temperature}

        {'Since this is in experiment mode, please start your response with "EXPERIMENT MODE [Temp: ' + str(st.session_state.temperature) + '] - " and include the similarity score and suggested response in your analysis.' if st.session_state.experiment_mode else 'Please provide a response to the user\'s question, taking into account the similar context and suggested response if they are relevant. If the similarity score is low, you may disregard the suggested context and response.'}"""

        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        
        try:
            response = groq_chat.invoke(st.session_state.conversation_history + [{"role": "user", "content": prompt}])
            ai_response = response.content

            st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

            st.text_area("AI's response:", value=ai_response, height=200, disabled=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if st.session_state.conversation_history:
    with st.expander("Show Previous Interactions"):
        for idx, interaction in enumerate(st.session_state.conversation_history):
            if interaction['role'] == 'user':
                st.markdown(f"**You:** {interaction['content']}")
            else:
                st.markdown(f"**LARMS:** {interaction['content']}")
            if idx < len(st.session_state.conversation_history) - 1:
                st.markdown("---")