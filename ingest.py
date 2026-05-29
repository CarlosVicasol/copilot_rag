import os
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
import config

def load_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == ".docx":
        doc = Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def run_ingestion(model):
    client = chromadb.PersistentClient(path=config.VECTORDB_PATH)
    
    # Intentamos borrar la colección existente para asegurar una indexación limpia
    try:
        client.delete_collection(name="document_collection")
    except Exception:
        pass
        
    collection = client.get_or_create_collection(name="document_collection")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    # Opcional: Limpiar colección antes de re-indexar para evitar basura
    # collection.delete(where={"source": filename}) # Necesitarías hacerlo por archivo

    for filename in os.listdir(config.DATA_PATH):
        file_path = os.path.join(config.DATA_PATH, filename)
        content = load_text_from_file(file_path)
        if content:
            # Eliminar fragmentos antiguos de este archivo específico para evitar duplicados
            collection.delete(where={"source": filename})
            chunks = split_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": filename})
                all_ids.append(f"{filename}_{i}")

    if all_chunks:
        # Usamos batch_size y mostramos progreso para mejorar rendimiento y feedback
        embeddings = model.encode(
            all_chunks, 
            batch_size=config.BATCH_SIZE, 
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        collection.upsert(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas)
    
    return f"Se han indexado {len(all_chunks)} fragmentos."