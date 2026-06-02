# Academic PDF RAG Agent

Academic PDF RAG Agent is a beginner-friendly Streamlit app that lets users upload one academic PDF and ask questions about its content. The app extracts text from the PDF, creates semantic embeddings, stores them in a ChromaDB vector collection, retrieves relevant evidence, and generates an answer grounded only in the uploaded document.

## Features

- Upload one PDF file
- Extract text from each PDF page with PyMuPDF
- Split page text into chunks with page and chunk metadata
- Generate semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store and search document chunks with ChromaDB
- Ask natural-language questions about the PDF
- Generate a fallback evidence-based answer without any API key
- Optionally use OpenAI or Gemini when API keys are available
- Show source page numbers and expandable retrieved evidence
- Display vector distances and similarity-style scores for retrieved chunks
- Show an Agent Workflow panel for question analysis, retrieval, evidence checks, answer generation, and citation
- Label retrieved evidence as Strong, Medium, or Weak before answering

## Tech Stack

- Python
- Streamlit for the web interface
- PyMuPDF for PDF text extraction
- sentence-transformers for embeddings
- ChromaDB for vector search
- NumPy and pandas for supporting data work

## How the RAG Workflow Works

1. The user uploads a single PDF.
2. PyMuPDF extracts text from each page.
3. The extracted text is split into overlapping chunks.
4. Each chunk keeps its text, page number, and chunk id.
5. `sentence-transformers/all-MiniLM-L6-v2` converts chunks into embeddings.
6. ChromaDB stores the chunk text, metadata, and embeddings in a vector collection.
7. When the user asks a question, the app embeds the question with the same model.
8. ChromaDB retrieves the most relevant chunks using vector similarity.
9. The answer generator uses only the retrieved evidence.
10. The app shows the answer, source pages, and expandable evidence sections.

## Agent Workflow

After each question, the app follows a simple agent-style workflow:

1. Question Understanding: extract important keywords from the user question.
2. Semantic Retrieval: embed the question and retrieve relevant PDF chunks from ChromaDB.
3. Evidence Quality Check: compare semantic similarity and keyword overlap to label the evidence as Strong, Medium, or Weak.
4. Grounded Answer Generation: answer only from retrieved PDF evidence, or refuse when the evidence is weak.
5. Source Citation: show the page numbers used by the final answer.

## Why This Is an AI Agent Project

This project is not only a search app. It follows a controlled reasoning workflow before answering:

- It understands the question by extracting keywords.
- It retrieves semantically relevant PDF evidence.
- It checks evidence quality before generating an answer.
- It refuses or becomes cautious when the evidence is weak.
- It grounds answers in retrieved PDF evidence.
- It provides source pages so users can verify the answer.

## How to Run Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/Lovis-Ghost/academic-pdf-rag-agent.git
cd academic-pdf-rag-agent
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Optional API Keys

The app works without an API key by using the built-in fallback answer generator. The fallback answer selects the most relevant sentences from retrieved evidence and includes this note:

```text
This answer is based only on the retrieved PDF evidence.
```

Optional LLM support can be enabled with environment variables or Streamlit secrets:

```bash
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

For Streamlit secrets, create `.streamlit/secrets.toml` locally. This file is ignored by git.

```toml
OPENAI_API_KEY = "your-key"
GEMINI_API_KEY = "your-key"
```

The optional providers also require their matching Python packages if you choose to use them:

```bash
pip install openai google-generativeai
```

## Example Questions

- What is the main argument of this paper?
- How does the author define the key concept?
- What method does the study use?
- What are the main findings?
- What limitations are mentioned?
- Which page discusses the experiment results?

## Limitations

- This version supports one PDF at a time.
- It does not support scanned PDFs unless they already contain selectable text.
- The fallback answer generator is simple and extractive.
- Retrieval quality depends on the PDF text quality and chunking.
- Optional LLM answers still depend entirely on the retrieved chunks.

## Future Improvements

- Add OCR support for scanned PDFs
- Improve chunking with sentence-aware splitting
- Add persistent vector storage
- Add multi-PDF support
- Add better citation formatting
- Add evaluation examples for retrieval quality

## Project Structure

```text
academic-pdf-rag-agent
├── app.py
├── rag_utils.py
├── README.md
├── requirements.txt
└── .gitignore
```
