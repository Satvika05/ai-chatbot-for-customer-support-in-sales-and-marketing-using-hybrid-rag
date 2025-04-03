import streamlit as st
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load intents data
with open(r"C:\Users\mysel\OneDrive\Pictures\Screenshots\AI_CHATBOT\intents.json", "r") as file:
    intents = json.load(file)

# Load SentenceTransformer model for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast model

# Prepare data for FAISS indexing
responses = []
corpus_embeddings = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        responses.append(intent["responses"][0])  # Take first response
        corpus_embeddings.append(embed_model.encode(pattern, convert_to_numpy=True))

# Convert to NumPy array
corpus_embeddings = np.array(corpus_embeddings, dtype=np.float32)

# Create FAISS index
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])  # L2 distance (Euclidean)
index.add(corpus_embeddings)

# Load GPT model (or OpenAI API)
gpt_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

# Function to retrieve response
def retrieve_response(query):
    query_embedding = embed_model.encode(query, convert_to_numpy=True).reshape(1, -1)
    _, I = index.search(query_embedding, k=1)  # Find closest match
    return responses[I[0][0]]  # Return best-matching response

# Function to generate final response
def chatbot_response(user_input):
    retrieved_text = retrieve_response(user_input)
    prompt = f"Customer asked: {user_input}\nSales support responded: {retrieved_text}\nEnhance this response naturally."
    
    # Generate response using GPT model
    generated_response = gpt_pipeline(prompt, max_length=100, do_sample=True)[0]["generated_text"]
    
    return generated_response

# Streamlit UI
st.title("🤖 AI Chatbot for Customer Support (Hybrid RAG)")

st.markdown("""
### 💬 Ask me anything about Sales & Marketing!
This AI chatbot retrieves the best response using **FAISS** and enhances it using **GPT**.
""")

# Chat input
user_query = st.text_input("Type your question:")

if st.button("Ask AI"):
    if user_query:
        response = chatbot_response(user_query)
        st.success(response)
    else:
        st.warning("Please enter a query.")
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/streamlit-chatbot.git
git push -u origin main
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/streamlit-chatbot.git
git push -u origin main
