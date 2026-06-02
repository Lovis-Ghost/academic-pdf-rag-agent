import streamlit as st

from rag_utils import (
    EMBEDDING_MODEL_NAME,
    build_vector_store,
    create_chunks,
    extract_pages_from_pdf,
    generate_answer,
    load_embedding_model,
    retrieve_chunks,
)


st.set_page_config(
    page_title="Academic PDF RAG Agent",
    page_icon="📚",
    layout="wide",
)


def reset_pdf_state():
    st.session_state.pop("pdf_name", None)
    st.session_state.pop("chunks", None)
    st.session_state.pop("collection", None)


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    answer_provider = st.selectbox(
        "Answer generator",
        ["Fallback only", "OpenAI", "Gemini"],
        help=(
            "OpenAI and Gemini require API keys in environment variables or "
            "Streamlit secrets. The fallback works without any API key."
        ),
    )

    st.header("RAG Workflow")
    st.write("1. Extract page text with PyMuPDF")
    st.write("2. Split pages into chunks with page metadata")
    st.write("3. Embed chunks with MiniLM")
    st.write("4. Store and search vectors with ChromaDB")
    st.write("5. Answer only from retrieved evidence")

st.title("Academic PDF RAG Agent")
st.write(
    "Upload one academic PDF, ask a question, and get an evidence-based answer "
    "with source page citations."
)

uploaded_file = st.file_uploader(
    "Upload one PDF",
    type=["pdf"],
    on_change=reset_pdf_state,
)

if uploaded_file is None:
    st.info("Upload a PDF to build the semantic search index.")
    st.stop()

if st.session_state.get("pdf_name") != uploaded_file.name:
    reset_pdf_state()

    with st.spinner("Extracting pages, creating chunks, and building the vector index..."):
        pages = extract_pages_from_pdf(uploaded_file)
        chunks = create_chunks(pages)

        if not chunks:
            st.error("No readable text was found in this PDF.")
            st.stop()

        embedding_model = load_embedding_model()
        collection = build_vector_store(chunks, embedding_model)

        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chunks = chunks
        st.session_state.collection = collection

chunks = st.session_state.chunks
collection = st.session_state.collection

st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")
st.caption(f"Embedding model: {EMBEDDING_MODEL_NAME}")

question = st.text_input("Ask a question about the uploaded PDF")

if not question:
    st.stop()

embedding_model = load_embedding_model()
retrieved_chunks = retrieve_chunks(
    collection=collection,
    embedding_model=embedding_model,
    question=question,
    top_k=top_k,
)

answer = generate_answer(
    question=question,
    retrieved_chunks=retrieved_chunks,
    provider=answer_provider,
)

st.subheader("Answer")
st.write(answer)

if retrieved_chunks:
    source_pages = sorted({chunk["page"] for chunk in retrieved_chunks})
    st.info(
        "Retrieved source page(s): "
        + ", ".join(str(page) for page in source_pages)
    )

st.subheader("Retrieved Evidence")

if not retrieved_chunks:
    st.write("No evidence was retrieved.")
else:
    for index, chunk in enumerate(retrieved_chunks, start=1):
        similarity = 1 - chunk["distance"]
        with st.expander(
            (
                f"Evidence {index} | Page {chunk['page']} | "
                f"Chunk {chunk['chunk_id']} | Distance {chunk['distance']:.4f} | "
                f"Similarity {similarity:.4f}"
            )
        ):
            st.write(chunk["text"])
