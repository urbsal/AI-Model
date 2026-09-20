"""
Document Similarity Search App
--------------------------------
Combines:
  1) Sample documents already sitting in the local "documents/" folder
  2) Files the user uploads at runtime (pdf, docx, txt)

Builds a single FAISS index over all chunks and lets the user search
for the most similar chunks to a typed query.

Run with:
    streamlit run app.py
"""

import os
import io
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

FOLDER_PATH = "documents/"
CHUNK_SIZE = 300  # words per chunk


# ----------------------------------------------------------------------
# Text extraction
#
# Same idea as a plain "file_path.endswith(...)" check, except an
# uploaded file has no real path on disk -- just a filename and an
# in-memory file object -- so we check the name string and read from
# the object itself instead of re-opening a path.
# ----------------------------------------------------------------------
def extract_text(filename, file_obj) -> str:
    text = ""

    if filename.endswith(".pdf"):
        reader = PdfReader(file_obj)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    elif filename.endswith(".docx"):
        doc = Document(file_obj)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif filename.endswith(".txt"):
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        text = raw

    return text.strip()


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def text_breakdown(text: str, chunk_size: int = CHUNK_SIZE):
    words = text.split()
    pieces = []
    for i in range(0, len(words), chunk_size):
        piece = words[i:i + chunk_size]
        pieces.append(" ".join(piece))
    return pieces


# ----------------------------------------------------------------------
# Loading: sample files on disk + files the user uploaded
# ----------------------------------------------------------------------
def load_all_documents(uploaded_files):
    documents, sources = [], []

    # sample files already sitting in documents/
    if os.path.isdir(FOLDER_PATH):
        for file in os.listdir(FOLDER_PATH):
            if file.endswith((".pdf", ".docx", ".txt")):
                path = os.path.join(FOLDER_PATH, file)
                with open(path, "rb") as f:
                    content = extract_text(file, f)
                chunks = text_breakdown(content)
                documents.extend(chunks)
                sources.extend([file] * len(chunks))

    # files the user just uploaded through the UI
    for uf in uploaded_files or []:
        if uf.name.endswith((".pdf", ".docx", ".txt")):
            content = extract_text(uf.name, uf)
            chunks = text_breakdown(content)
            documents.extend(chunks)
            sources.extend([uf.name] * len(chunks))

    return documents, sources


# ----------------------------------------------------------------------
# Model (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------------------------------------------
# Index building
# ----------------------------------------------------------------------
def build_index(documents):
    model = get_model()
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search(query, model, index, documents, sources, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, ids = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "source": sources[idx],
            "text": documents[idx],
        })
    return results


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Document Similarity Search", layout="wide")
    st.title("📄 Document Similarity Search")
    st.caption(
        "Searches across sample files in the local `documents/` folder "
        "plus any files you upload below."
    )

    uploaded_files = st.file_uploader(
        "Upload additional files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    rebuild = st.button("Build / Rebuild Index", type="primary")

    # Rebuild the index whenever the button is pressed, or on first load.
    if rebuild or "index" not in st.session_state:
        with st.spinner("Reading files and building index..."):
            documents, sources = load_all_documents(uploaded_files)

            if not documents:
                st.warning(
                    "No readable content found. Add files to the `documents/` "
                    "folder or upload some, then rebuild."
                )
                st.session_state.pop("index", None)
            else:
                index = build_index(documents)
                st.session_state["index"] = index
                st.session_state["documents"] = documents
                st.session_state["sources"] = sources
                st.success(
                    f"Index built: {len(documents)} chunks from "
                    f"{len(set(sources))} file(s)."
                )

    st.divider()

    if "index" in st.session_state:
        query = st.text_input("Enter your search query")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        if query:
            model = get_model()
            results = search(
                query,
                model,
                st.session_state["index"],
                st.session_state["documents"],
                st.session_state["sources"],
                top_k=top_k,
            )

            if not results:
                st.info("No results found.")
            else:
                for i, r in enumerate(results, start=1):
                    with st.container(border=True):
                        st.markdown(f"**#{i} — {r['source']}**  (score: {r['score']:.3f})")
                        st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
    else:
        st.info("Build the index above to start searching.")


if __name__ == "__main__":
    main()