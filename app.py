import streamlit as st
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Documents")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------- FILE READING --------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt(file):
    return file.read().decode("utf-8")

# -------- CHUNKING --------
def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# -------- BUILD INDEX --------
def build_index(docs):
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, chunks

# -------- RETRIEVE --------
def retrieve(query, index, chunks):
    q = model.encode([query])
    _, idx = index.search(np.array(q), 3)
    return [chunks[i] for i in idx[0]]

# -------- UI UPLOAD --------
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

docs = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            docs.append(read_pdf(file))
        else:
            docs.append(read_txt(file))
else:
    docs = ["Default document about AI"]

# -------- BUILD INDEX DYNAMIC --------
if "index" not in st.session_state or uploaded_files:
    with st.spinner("Processing documents..."):
        index, chunks = build_index(docs)
        st.session_state.index = index
        st.session_state.chunks = chunks

# -------- QUERY --------
query = st.text_input("Ask a question:")

if query:
    start = time.time()

    context_chunks = retrieve(query, st.session_state.index, st.session_state.chunks)
    context = " ".join(context_chunks)

    answer = "Answer: " + context[:300]

    end = time.time()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Response Time")
    st.write(f"{end - start:.2f} sec")

    with st.expander("Context"):
        for i, c in enumerate(context_chunks):
            st.write(f"{i+1}. {c}")
