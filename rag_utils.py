import os
import re
import uuid

import chromadb
import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("�", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages_from_pdf(uploaded_file):
    pages = []
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    for page_index in range(len(pdf_document)):
        text = clean_text(pdf_document[page_index].get_text())
        if text:
            pages.append({
                "page": page_index + 1,
                "text": text,
            })

    pdf_document.close()
    return pages


def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(pages, chunk_size=900, overlap=150):
    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "id": f"page-{page['page']}-chunk-{chunk_index}",
                "page": page["page"],
                "chunk_id": chunk_index,
                "text": chunk,
            })

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model, texts):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32).tolist()


def build_vector_store(chunks, embedding_model):
    client = chromadb.Client()
    collection_name = f"pdf_chunks_{uuid.uuid4().hex}"
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(embedding_model, texts)

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ],
    )

    return collection


def retrieve_chunks(collection, embedding_model, question, top_k=4):
    question_embedding = embed_texts(embedding_model, [question])[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "text": document,
            "page": metadata["page"],
            "chunk_id": metadata["chunk_id"],
            "distance": float(distance),
        })

    return retrieved


def format_evidence_for_prompt(retrieved_chunks):
    evidence_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        evidence_blocks.append(
            f"[Evidence {index} | Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(evidence_blocks)


def get_secret_or_env(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)


def generate_fallback_answer(retrieved_chunks, max_chunks=3, max_chars=1800):
    if not retrieved_chunks:
        return (
            "The answer cannot be found from the uploaded PDF because no relevant "
            "evidence was retrieved."
        )

    best_distance = retrieved_chunks[0]["distance"]
    if best_distance > 0.85:
        return (
            "The answer cannot be found confidently from the uploaded PDF. "
            "The retrieved evidence appears weak or only loosely related."
        )

    selected = retrieved_chunks[:max_chunks]
    combined = " ".join(chunk["text"] for chunk in selected)
    combined = combined[:max_chars].strip()

    pages = sorted({chunk["page"] for chunk in selected})
    page_text = ", ".join(str(page) for page in pages)

    return (
        "This answer is based only on the retrieved PDF evidence.\n\n"
        f"Relevant evidence from page(s) {page_text}: {combined}"
    )


def generate_openai_answer(question, retrieved_chunks):
    api_key = get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    evidence = format_evidence_for_prompt(retrieved_chunks)
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=get_secret_or_env("OPENAI_MODEL") or "gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer academic PDF questions using only the provided "
                    "evidence. If the evidence does not contain the answer, say so. "
                    "Cite page numbers from the evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved PDF evidence:\n{evidence}\n\n"
                    "Answer with a short, clear response and page citations."
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def generate_gemini_answer(question, retrieved_chunks):
    api_key = get_secret_or_env("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        return None

    evidence = format_evidence_for_prompt(retrieved_chunks)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(get_secret_or_env("GEMINI_MODEL") or "gemini-1.5-flash")

    prompt = (
        "Answer the academic PDF question using only the provided evidence. "
        "If the evidence does not contain the answer, say so. Cite page numbers.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved PDF evidence:\n{evidence}"
    )
    response = model.generate_content(prompt)
    return response.text


def generate_answer(question, retrieved_chunks, provider="Fallback only"):
    if provider == "OpenAI":
        answer = generate_openai_answer(question, retrieved_chunks)
        if answer:
            return answer

    if provider == "Gemini":
        answer = generate_gemini_answer(question, retrieved_chunks)
        if answer:
            return answer

    return generate_fallback_answer(retrieved_chunks)
