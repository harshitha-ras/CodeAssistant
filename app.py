# Import libraries
import os
import uuid
import openai
import chromadb
import streamlit as st
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from huggingface_hub import login

# Authenticate APIs using secrets from Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]
login(token=st.secrets["HF_TOKEN"])

# Load dataset (with error handling)
try:
    ds = load_dataset("bigcode/the-stack", split="train", streaming=True, revision="main")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")

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

# Process dataset into chunks (limit examples dynamically)
num_examples = st.slider("Number of examples to process", min_value=10, max_value=100, step=10)
chunks = []
for example in ds.take(num_examples):
    code = example.get('content', '')  # Check dataset structure if 'content' is invalid
    if code:
        chunks.extend(chunk_text(code))

# Initialize ChromaDB with SQLite backend
client = chromadb.Client(Settings(
    chroma_db_impl="sqlite",
    persist_directory=".chromadb"
))
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

# Streamlit interface for user queries
st.title("Coding Assistant")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating response..."):
        try:
            # Perform semantic search on ChromaDB collection
            query_embedding = model.encode(query)
            results = collection.query(query_embeddings=[query_embedding], n_results=5)

            # Prepare context for LLM prompt (handle missing documents)
            if "documents" in results:
                context = "\n".join(results["documents"])
                prompt = f"Context:\n{context}\n\nQuestion: {query}"

                # Generate response using OpenAI's GPT model (text-davinci-003)
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7,
                )
                st.write(response["choices"][0]["text"])
            else:
                st.error("No relevant documents found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

import streamlit as st
st.write(st.secrets)
