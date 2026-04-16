import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# ── CONFIG ──────────────────────────────────────────
DOCS_FOLDER   = "docs"
CHUNK_SIZE    = 200   # characters per chunk
CHUNK_OVERLAP = 40    # overlap between chunks
TOP_K         = 3     # how many chunks to retrieve
MODEL_NAME    = "llama3"
# ────────────────────────────────────────────────────


def load_documents(folder):
    """Read all .txt files from the docs folder."""
    all_text = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r") as f:
                text = f.read()
                all_text.append((filename, text))
            print(f"  Loaded: {filename}")
    return all_text


def chunk_text(filename, text, chunk_size, overlap):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": filename,
                "start": start
            })
        start += chunk_size - overlap
    return chunks


def build_index(chunks, embed_model):
    """Embed all chunks and store in FAISS."""
    texts = [c["text"] for c in chunks]
    print(f"\n  Embedding {len(texts)} chunks...")
    vectors = embed_model.encode(texts, show_progress_bar=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index, vectors


def retrieve(query, index, chunks, embed_model, k):
    """Find top-k most relevant chunks for the query."""
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    results = []
    for i in I[0]:
        if i < len(chunks):
            results.append(chunks[i])
    return results


def generate_answer(query, context_chunks):
    """Send query + retrieved context to LLM."""
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks
    ])
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(model=MODEL_NAME, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']


# ── MAIN ────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   Mini RAG Document Query System")
    print("=" * 50)

    # 1. Load documents
    print("\n[1] Loading documents...")
    documents = load_documents(DOCS_FOLDER)
    if not documents:
        print("No .txt files found in docs/ folder. Add some and retry.")
        return

    # 2. Chunk documents
    print("\n[2] Chunking documents...")
    all_chunks = []
    for filename, text in documents:
        chunks = chunk_text(filename, text, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        print(f"  {filename} → {len(chunks)} chunks")

    # 3. Load embedding model + build FAISS index
    print("\n[3] Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, _ = build_index(all_chunks, embed_model)
    print("  Index ready!")

    # 4. Query loop
    print("\n[4] Ready! Ask questions about your documents.")
    print("    Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == 'quit':
            print("Bye!")
            break

        # Retrieve relevant chunks
        top_chunks = retrieve(query, index, all_chunks, embed_model, TOP_K)

        print("\n--- Retrieved Context ---")
        for i, c in enumerate(top_chunks):
            print(f"  [{i+1}] ({c['source']}) {c['text'][:80]}...")

        # Generate LLM answer
        print("\n--- Answer ---")
        answer = generate_answer(query, top_chunks)
        print(f"{answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
