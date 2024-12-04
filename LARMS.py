import streamlit as st
import torch
import pandas as pd
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util

# Inject custom CSS to ensure full styling
st.markdown(
    """
    <style>
    /* Main App Background and Text */
    .stApp {
        background-color: #ffffff !important;
        color: black !important;
    }

    /* Sidebar Background and Text */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        color: black !important;
        border-right: 1px solid #e9ecef !important;
    }
    section[data-testid="stSidebar"] .css-17eq0hr {
        color: black !important;
    }

    /* Header (Deploy Section) Background and Text */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: black !important;
    }

    /* Make all titles and headers bold and black */
    h1, h2, h3, h4, h5, h6, .stTitle, .stHeader h1, .stHeader h2 {
        color: black !important;
        font-weight: 600 !important;
    }

    /* Specifically target the main title */
    .stTitle, [data-testid="stMarkdownContainer"] h1 {
        color: black !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }

    /* Make sidebar headers more prominent */
    .sidebar .stTitle, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: black !important;
        font-weight: 600 !important;
    }

    /* Style text input and text areas */
    .stTextInput input, .stTextArea textarea {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        font-weight: 500 !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        padding: 12px !important;
    }

    /* Force black text color for all textareas including disabled ones */
    textarea {
        color: #000000 !important;
    }

    /* Style disabled text areas (AI responses) */
    .stTextArea textarea:disabled {
        background-color: #f1f3f5 !important;
        color: #000000 !important;
        opacity: 1 !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }

    /* Add hover effect to text areas */
    .stTextInput input:hover, .stTextArea textarea:hover {
        background-color: #f1f3f5 !important;
        border-color: #ced4da !important;
        transition: all 0.3s ease !important;
    }

    /* Ensure all text content is black */
    .css-1v3fvcr, .css-10trblm, .css-1lh0qv1, 
    .css-1d391kg, .css-1lndpgh, .css-qbe2hs, 
    .css-1uxlu9e, .css-1u4a9nw {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Style buttons */
    .stButton button {
        color: black !important;
        font-weight: 500 !important;
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        background-color: #f1f3f5 !important;
        border-color: #ced4da !important;
    }

    /* Style markdown text */
    .element-container, .stMarkdown {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Custom styling for conversation history */
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, #f8f9fa, #ffffff) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        margin: 1rem 0 !important;
    }

    /* History expander header */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        color: black !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }

    /* History content */
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1rem !important;
    }

    /* Individual message styling in history */
    .streamlit-expanderContent p {
        padding: 0.8rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 8px !important;
        background-color: #f8f9fa !important;
        border-left: 4px solid #4a90e2 !important;
    }

    /* User message styling */
    .streamlit-expanderContent p strong:contains("You:") {
        color: #2c5282 !important;
    }

    /* AI message styling */
    .streamlit-expanderContent p strong:contains("LARMS:") {
        color: #38a169 !important;
    }

    /* Style slider */
    .stSlider {
        background-color: #ffffff !important;
        padding: 10px !important;
        border-radius: 8px !important;
        border: 1px solid #e9ecef !important;
    }

    /* Style toggle switch */
    .stCheckbox {
        background-color: #f8f9fa !important;
        padding: 10px !important;
        border-radius: 8px !important;
        border: 1px solid #e9ecef !important;
    }

    /* Style dataframe/table */
    .dataframe {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
    }

    /* Style select boxes */
    .stSelectbox {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
    }

    /* Add padding to containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Customize scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f3f5;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: #ced4da;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #adb5bd;
    }
        /* Force black text color for ALL textareas and their content */
    textarea, .stTextArea textarea, div[data-baseweb="textarea"] textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Specifically target the response textarea */
    .stTextArea div[data-baseweb="textarea"] textarea,
    .stTextArea textarea[disabled],
    .stTextArea textarea:disabled {
        background-color: #f1f3f5 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }

    /* Additional override for any possible default styles */
    .stTextArea textarea::placeholder,
    .stTextArea textarea:disabled::placeholder {
        color: #6c757d !important;
        -webkit-text-fill-color: #6c757d !important;
    }

    /* Override any webkit autofill styles */
    .stTextArea textarea:-webkit-autofill,
    .stTextArea textarea:-webkit-autofill:hover,
    .stTextArea textarea:-webkit-autofill:focus {
        -webkit-text-fill-color: #000000 !important;
        transition: background-color 5000s ease-in-out 0s;
    }

    /* Fix for Temperature and other labels */
    .stMarkdown div p, .stMarkdown div span {
        color: black !important;
    }

    /* Target all text elements in the app */
    div:not(.streamlit-expanderContent) p,
    div span,
    label span,
    .stMarkdown p,
    .stMarkdown span,
    .stSlider p,
    .stSlider span {
        color: black !important;
    }

    /* Target specifically the temperature value */
    .stSlider [data-testid="stMarkdownContainer"] p {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Experiment mode and other toggle text */
    .stCheckbox label span {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Fix for all input labels */
    .stTextInput label,
    .stTextArea label,
    .stSlider label {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Target any remaining text elements */
    [data-baseweb="input"] div,
    [data-baseweb="textarea"] div,
    [data-testid="stText"] p {
        color: black !important;
    }

    /* Force all text inputs and areas to be black */
    input, textarea {
        color: black !important;
        -webkit-text-fill-color: black !important;
    }

    /* Ensure experiment mode text is black */
    .stText, .stText p {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Additional styling for better visibility */
    .stApp label, .stApp span {
        color: black !important;
    }
   /* Style toggle/checkbox - General container */
    .stCheckbox {
        background-color: #f8f9fa !important;
        padding: 10px !important;
        border-radius: 8px !important;
        border: 1px solid #e9ecef !important;
    }

    /* Style the toggle switch text and background */
    .stCheckbox > label {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Style the toggle switch container */
    .stCheckbox > div[role="checkbox"] {
        background-color: #e9ecef !important;
    }

    /* Style the toggle switch when checked */
    .stCheckbox > div[role="checkbox"][aria-checked="true"] {
        background-color: #4a90e2 !important;
    }

    /* Style the toggle switch knob */
    .stCheckbox > div[role="checkbox"]::before {
        background-color: white !important;
    }

    /* Additional specific styling for experiment mode toggle */
    [data-testid="stCheckbox"] {
        background-color: #f8f9fa !important;
    }

    [data-testid="stCheckbox"] > label > div {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Force black text color for toggle label */
    div[data-testid="stCheckbox"] label span {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
    st.session_state.temperature = 0.75

df = pd.read_csv(merged_path)
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

with st.spinner("Loading dataset..."):
    df = pd.read_csv(merged_path,  low_memory=False)

groq_api_key = "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"

if groq_api_key:
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.2-90b-vision-preview",
        temperature=st.session_state.temperature  # Use the temperature from the slider
    )

    user_question = st.text_area("Is there something you want to share with me?")

if user_question:
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
    prompt = f"""You are an AI Powered Chatbot who provide remedies to queries, your remedies should always be confident and never sound lacking. Always sound emotionally strong and give confidence to the person that the remedy you provide definitely works. You should not respond to any other kind of questions which are unrelated to mental health and life.

    User question: {user_question}
    Similar context from database: {similar_context}
    Suggested response: {similar_response}
    Similarity score: {similarity_score}
    Current temperature: {st.session_state.temperature}

    {'Since this is in experiment mode, please start your response with "EXPERIMENT MODE [Temp: ' + str(st.session_state.temperature) + '] - " and include the similarity score and suggested response in your analysis.' if st.session_state.experiment_mode else 'Please provide a response to the user\'s question, taking into account the similar context and suggested response if they are relevant. If the similarity score is low, you may disregard the suggested context and response.'}"""

    # Add user input to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_question})

    # Generate the AI response
    with st.spinner("Generating AI response..."):
        try:
            response = groq_chat.invoke(st.session_state.conversation_history + [{"role": "user", "content": prompt}])
            ai_response = response.content

            # Add AI response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

            # Display AI response
            st.text_area("AI's response:", value=ai_response, height=200, disabled=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display previous interactions
if st.session_state.conversation_history:
    with st.expander("Show Previous Interactions"):
        for idx, interaction in enumerate(st.session_state.conversation_history):
            if interaction['role'] == 'user':
                st.markdown(f"**You:** {interaction['content']}")
            else:
                st.markdown(f"**LARMS:** {interaction['content']}")
            if idx < len(st.session_state.conversation_history) - 1:
                st.markdown("---")
