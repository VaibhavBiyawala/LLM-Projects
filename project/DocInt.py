import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a sentence-transformer model
embedder = SentenceTransformer("sheldonrobinson / Phi-3.5-vision-instruct")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to split text into smaller chunks
def split_text(text, chunk_size=30):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Load and process the PDF
pdf_path = input("Enter a file path : ")  
pdf_text = extract_text_from_pdf(pdf_path)
chunks = split_text(pdf_text)

# Convert chunks to vector embeddings
chunk_vectors = embedder.encode(chunks)

# Create FAISS index
dimension = chunk_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_vectors))

# Search function
def search_document(query, top_k=1):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Chat loop
while True:
    query = input("\nEnter your search query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    results = search_document(query)
    print("\nTop results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.strip()}")
