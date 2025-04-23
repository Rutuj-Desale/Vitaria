from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
# from pinecone import Pinecone  # Commented out for now
# from nomic import embed  # Commented out for now

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["vitaria"]
chat_collection = mongo_db["chat_history"]

# Commented: Initialize Pinecone
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# if not pinecone_api_key:
#     raise ValueError("PINECONE_API_KEY is not set in environment variables.")
# pc = Pinecone(api_key=pinecone_api_key)
# index = pc.Index("vitaria-memory")

# Commented: Function to generate embeddings
# def generate_embedding(text):
#     return embed.text([text], model="nomic-embed-text-v1.5")["embeddings"][0]

# Define API request structure
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Summarize long conversation text before storing (still used in future phase)
def summarize_conversation(conversation_text):
    if len(conversation_text.split()) > 150:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Summarize the following conversation in under 100 words without losing the important details:\n{conversation_text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    return conversation_text

# Commented: Store conversation in Pinecone
# def store_conversation_in_pinecone(user_id, user_message, bot_response):
#     conversation_entry = {
#         "user_id": user_id,
#         "user_message": user_message,
#         "bot_response": bot_response,
#         "timestamp": str(datetime.utcnow())
#     }
#     embedding = generate_embedding(user_message)
#     index.upsert([(user_id, embedding, {"user_message": user_message, "bot_response": bot_response})])

# Commented: Retrieve context from Pinecone
# def retrieve_context_from_pinecone(user_id, query):
#     query_embedding = generate_embedding(query)
#     results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
#     if results.get("matches"):
#         most_relevant_match = results["matches"][0]["metadata"]
#         return most_relevant_match.get("bot_response", "")
#     return ""

# Save conversation in MongoDB
def save_to_mongo(user_id, user_message, bot_response):
    chat_collection.insert_one({
        "user_id": user_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.utcnow()
    })

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Fetch last 5 chat messages from MongoDB for context
        past_chats = chat_collection.find({"user_id": request.user_id}).sort("timestamp", -1).limit(5)
        past_conversation = ""
        for chat in reversed(list(past_chats)):  # Reverse so oldest comes first
            past_conversation += f"User: {chat['user_message']}\nMedical Assistant: {chat['bot_response']}\n"

        # New user message
        full_prompt = f"""
You are a helpful and reliable medical chatbot assistant. Your job is to provide medically accurate, easy-to-understand, and trustworthy responses based on user input. Avoid generic statements, and always try to be informative and empathetic.

Here is the past conversation:
{past_conversation}
User: {request.message}
Medical Assistant:"""

        response = model.generate_content(full_prompt)
        final_response = response.text.strip()

        # Store new chat in MongoDB
        save_to_mongo(request.user_id, request.message, final_response)

        return {"response": final_response}
    except Exception as e:
        return {"error": str(e)}
