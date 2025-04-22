from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
from nomic import embed

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("vitaria-memory")

# Function to generate embeddings using a free model
def generate_embedding(text):
    return embed.text([text], model="nomic-embed-text-v1.5")["embeddings"][0]

# Define API request structure
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Summarize long conversation text before storing
def summarize_conversation(conversation_text):
    if len(conversation_text.split()) > 150:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Summarize the following conversation in under 100 words without losing the important details:\n{conversation_text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    return conversation_text

# Store conversation in Pinecone
def store_conversation_in_pinecone(user_id, user_message, bot_response):
    conversation_entry = {
        "user_id": user_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": str(datetime.utcnow())
    }
    embedding = generate_embedding(user_message)  # Store embedding for user message only
    index.upsert([(user_id, embedding, {"user_message": user_message, "bot_response": bot_response})])

# Retrieve past context from Pinecone
def retrieve_context_from_pinecone(user_id, query):
    query_embedding = generate_embedding(query)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    if results.get("matches"):
        most_relevant_match = results["matches"][0]["metadata"]
        return most_relevant_match.get("bot_response", "")
    return ""

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Retrieve the most relevant past response
        context = retrieve_context_from_pinecone(request.user_id, request.message)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        if context:
            prompt = f"{context}\nUser: {request.message}\nBot:"
        else:
            prompt = request.message

        response = model.generate_content(prompt)
        final_response = response.text.strip()
        store_conversation_in_pinecone(request.user_id, request.message, final_response)
        
        return {"response": final_response}
    except Exception as e:
        return {"error": str(e)}