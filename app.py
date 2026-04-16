import streamlit as st
import requests
import numpy as np
import faiss
import os
import time
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import PyPDF2

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "llama3"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Documents")

def correct_query(query):
    try:
        return str(TextBlob(query).correct())
    except:
        return query

def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def process_uploaded_files(files):
    docs = []
    for file in files:
        if file.name.endswith(".txt"):
            docs.append(read_txt(file))
        elif file.name.endswith(".pdf"):
            docs.append(read_pdf(file))
    return docs

def load_docs():
    docs = []
    if os.path.exists("docs"):
        for file in os.listdir("docs"):
            with open(os.path.join("docs", file), "r") as f:
                docs.append(f.read())
    return docs

@st.cache_resource
def setup_rag(docs):
    if not docs:
        return None, None, []

    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = []
    for doc in docs:
        for i in range(0, len(doc), 500):
            chunks.append(doc[i:i+500])

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, chunks

def retrieve(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(context, query):
    prompt = f"""
You are a professional assistant.

Convert the context into a clean, human-friendly answer.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False}
    )

    return response.json()["response"]

uploaded_files = st.file_uploader("Upload your documents", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    docs = process_uploaded_files(uploaded_files)
else:
    docs = load_docs()

model, index, chunks = setup_rag(tuple(docs))

query = st.text_input("Ask a question:")

if query:
    query = correct_query(query)

    start = time.time()

    context_chunks = retrieve(query, model, index, chunks)
    context = " ".join(context_chunks)

    answer = "Demo Answer: " + context[:200]

    end = time.time()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Response Time")
    st.write(f"{end - start:.2f} seconds")

    with st.expander("Retrieved Context"):
        for i, chunk in enumerate(context_chunks):
            st.write(f"{i+1}. {chunk}")
