cat > app.py << 'EOF'
import streamlit as st
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Documents")

# ---------------- LOAD MODEL (CACHE) ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- READ FILES ----------------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt(file):
    return file.read().decode("utf-8")

# ---------------- CHUNKING ----------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# ---------------- BUILD INDEX ----------------
def build_index(docs):
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

# ---------------- RETRIEVE ----------------
def retrieve(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# ---------------- UI ----------------
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------- PROCESS DOCS ----------------
docs = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            docs.append(read_pdf(file))
        else:
            docs.append(read_txt(file))
else:
    docs = ["This is a default document about AI and machine learning."]

# ---------------- BUILD RAG ----------------
if "index" not in st.session_state:
    with st.spinner("Processing documents..."):
        index, chunks = build_index(docs)
        st.session_state.index = index
        st.session_state.chunks = chunks

# ---------------- QUERY ----------------
query = st.text_input("Ask a question:")

if query:
    start = time.time()

    context_chunks = retrieve(query, st.session_state.index, st.session_state.chunks)
    context = " ".join(context_chunks)

    # simple answer (no LLM to avoid crash)
    answer = "Answer based on documents: " + context[:300]

    end = time.time()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Response Time")
    st.write(f"{end - start:.2f} seconds")

    with st.expander("Retrieved Context"):
        for i, chunk in enumerate(context_chunks):
            st.write(f"{i+1}. {chunk}")
EOF
