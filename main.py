from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import os
from langchain_groq import ChatGroq
import dotenv
import pandas as pd

# Load environment variables
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file at the root URL
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Load the SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths for dataset and embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # or your specific path
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "corpus/embeddings.pt")
MERGED_PATH = os.path.join(BASE_DIR, "corpus/merged_dataset.csv")

# Ensure paths exist
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MERGED_PATH), exist_ok=True)

# Groq initialization function
def initialize_groq(api_key, model_name, temperature):
    try:
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq initialization error: {e}")

# Function to load or compute embeddings
def load_or_compute_embeddings(df):
    if os.path.exists(EMBEDDINGS_PATH):
        return torch.load(EMBEDDINGS_PATH)
    else:
        embeddings = embedding_model.encode(df["Context"].tolist(), convert_to_tensor=True)
        torch.save(embeddings, EMBEDDINGS_PATH)
        return embeddings

# Request model
class UserInput(BaseModel):
    user_question: str
    selected_model: str = "llama-3.2-90b-vision-preview"  # Default model
    temperature: float = 0.7  # Default temperature

@app.post("/chat")
async def chat_with_model(input_data: UserInput):
    try:
        # Load dataset and embeddings
        df = pd.read_csv(MERGED_PATH)
        contexts = df["Context"].tolist()
        responses = df["Response"].tolist()
        context_embeddings = load_or_compute_embeddings(df)

        # Find the most similar context
        user_embedding = embedding_model.encode(input_data.user_question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, context_embeddings)[0]
        max_idx = torch.argmax(similarities).item()
        similar_context = contexts[max_idx]
        similar_response = responses[max_idx]

        # Initialize Groq with the provided parameters
        groq_chat = initialize_groq(
            api_key=os.getenv("GROQ_API_KEY", "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"),
            model_name=input_data.selected_model,
            temperature=input_data.temperature
        )

        # Create the prompt for the Groq model
        prompt = f"""
        Your name is LARMS purposefully designed for Mental Health Conversations. Be Concise and make sure user question and context are similar. This is designed more like a messaging app and you have to talk like you are messaging
        This is the User question: {input_data.user_question}
        Context: {similar_context}
        Response: {similar_response}
        """

        # Get the response from Groq
        response = groq_chat.invoke([{"role": "user", "content": prompt}])
        if not response or not hasattr(response, 'content'):
            raise ValueError("No valid response received from Groq")

        ai_response = response.content

        # Return AI response
        return {"ai_response": ai_response}

    except Exception as e:
        # Log the error and return a detailed message
        print(f"Error occurred: {e}")  # For server-side debugging
        raise HTTPException(status_code=500, detail=f"Error processing chat response: {e}")
