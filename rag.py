from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import config

class DocumentMock:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def ask_rag(query, model, client, groq_client):
    # Acceder a la colección
    collection = client.get_or_create_collection(name="document_collection")
    
    # Búsqueda
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    
    relevant_docs = [DocumentMock(doc, meta) for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""Responde únicamente basándote en el contexto proporcionado.
    Si no encuentras la respuesta, indica que no está en los documentos.

    Contexto:
    {context}

    Pregunta:
    {query}

    Respuesta:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un asistente experto en documentos"},
            {"role": "user", "content": prompt}
        ],
        model=config.GROQ_MODEL,
        temperature=0.1 # Baja temperatura para mayor fidelidad al texto
    )
    
    return chat_completion.choices[0].message.content, relevant_docs