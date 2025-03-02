import os
import uuid
import openai
import chromadb
import streamlit as st
import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from chromadb.config import Settings
from chromadb import Client, PersistentClient
from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction



# Authenticate APIs using secrets from Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]
login(token=st.secrets["HF_TOKEN"], add_to_git_credential=True)

# Download the punkt tokenizer
nltk.download('punkt', quiet=True)

# Load dataset (with error handling)
@st.cache_resource
def load_dataset_cached():
    try:
        return load_dataset("bigcode/the-stack", split="train", streaming=True, revision="main", timeout=120)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

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

# Initialize a persistent client
@st.cache_resource
def get_chroma_client():
    return PersistentClient(path=".chromadb")

# Load embedding model (SentenceTransformer)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-V2')

# Streamlit interface
st.title("Coding Assistant")

# Initialize components
ds = load_dataset_cached()
client = get_chroma_client()
model = load_embedding_model()

# Create embedding function
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# Create or get collection
collection = client.get_or_create_collection("coding_assistant", embedding_function=sentence_transformer_ef)

# Process dataset into chunks
if ds:
    num_examples = st.slider("Number of examples to process", min_value=10, max_value=100, step=10)
    process_button = st.button("Process Examples")

    if process_button:
        with st.spinner("Processing examples..."):
            chunks = []
            for example in ds.take(num_examples):
                code = example.get('content', '')
                if code:
                    chunks.extend(chunk_text(code))

            # Add chunks to ChromaDB with embeddings and metadata
            collection.add(
                documents=chunks,
                metadatas=[{"source": "The Stack"} for _ in chunks],
                ids=[str(uuid.uuid4()) for _ in chunks]
            )
        st.success(f"Processed {len(chunks)} chunks from {num_examples} examples.")

# User query interface
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating response..."):
        try:
            # Perform semantic search on ChromaDB collection
            results = collection.query(query_texts=[query], n_results=5)

            # Prepare context for LLM prompt
            if results["documents"]:
                context = "\n".join(results["documents"][0])
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

                # Generate response using OpenAI's GPT model
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "prompt"}]
                )
                st.write(response.choices[0].text.strip())
            else:
                st.warning("No relevant documents found. Try processing more examples or rephrasing your question.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
