# Course PDF Q&A Assistant

A Streamlit-based PDF question answering assistant that allows users to upload course PDF files, ask questions, retrieve relevant evidence, and generate simple answers based on the document content.

## Project Overview

This project is designed to help students quickly understand course materials by asking questions directly from uploaded PDF documents.

The system extracts text from a PDF, retrieves the most relevant content using TF-IDF similarity, and generates a simple answer with source page references.

## Features

- Upload course PDF files
- Extract useful text blocks from PDF pages
- Ask questions based on the uploaded document
- Retrieve relevant evidence using TF-IDF similarity
- Generate a simple answer from retrieved content
- Display source page numbers
- Show expandable evidence sections
- Streamlit web interface

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## How It Works

1. The user uploads a course PDF.
2. The system extracts text blocks from the PDF using PyMuPDF.
3. The user enters a question.
4. The question and extracted text blocks are converted into TF-IDF vectors.
5. Cosine similarity is used to find the most relevant text blocks.
6. A simple answer is generated from the retrieved evidence.
7. The answer and source evidence are displayed in the web app.

## Project Structure

```text
course-pdf-qa-assistant
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── venv