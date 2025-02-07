import fitz  # PyMuPDF
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# Load model
embedder = SentenceTransformer("msmarco-distilbert-base-v4")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Split text into chunks
def split_text(text, chunk_size=100):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Load PDF & Build FAISS Index
def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(pdf_text)
    chunk_vectors = embedder.encode(chunks)

    # FAISS Index
    index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    index.add(np.array(chunk_vectors))
    return chunks, index

# Search Function
def search_document(query, chunks, index, top_k=2):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

# UI Function
def chatbot_ui(pdf_file, query):
    chunks, index = process_pdf(pdf_file.name)
    results = search_document(query, chunks, index)
    return "\n\n".join(results)

# Launch Gradio Interface
gr.Interface(fn=chatbot_ui,
             inputs=["file", "text"],
             outputs="text",
             title="📄 AI Document Search Chatbot",
             description="Upload a PDF & ask questions!").launch()
