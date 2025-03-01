# Import libraries
import os
import uuid
import openai
import chromadb
import streamlit as st
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Set up OpenAI API key (replace with your actual key)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load The Stack dataset (streaming mode)
ds = load_dataset("bigcode/the-stack", split="train", streaming=True)

# Chunking function
def chunk_text(text, max_tokens=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        tokens = len(sentence.split())
        if current_length + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(sentence)
        current_length += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Extract and chunk data from The Stack dataset (limit examples for testing)
chunks = []
for example in ds.take(100):  # Adjust the number of examples as needed
    code = example.get('content', '')  # Assuming 'content' contains the code snippets
    if code:
        chunks.extend(chunk_text(code))

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection("coding_assistant")

# Load embedding model (SentenceTransformer)
model = SentenceTransformer('all-MiniLM-L6-V2')

# Add chunks to ChromaDB with embeddings and metadata
for chunk in chunks:
    embedding = model.encode(chunk)
    collection.add(
        documents=[chunk],
        metadatas=[{"source": "The Stack"}],
        ids=[str(uuid.uuid4())]
    )

# Streamlit app interface for user queries
st.title("Coding Assistant")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating response..."):
        try:
            # Perform semantic search on ChromaDB collection
            query_embedding = model.encode(query)
            results = collection.query(query_embeddings=[query_embedding], n_results=5)

            # Prepare context for LLM prompt
            context = "\n".join(results['documents'])
            prompt = f"Context:\n{context}\n\nQuestion: {query}"

            # Generate response using OpenAI's GPT model (text-davinci-003)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
            )

            # Display response in Streamlit app
            st.write(response["choices"][0]["text"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
