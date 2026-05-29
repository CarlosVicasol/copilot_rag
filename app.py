import streamlit as st
import os
import config
from ingest import run_ingestion
from rag import ask_rag
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import logging

# Configuración de logs para evitar ruido en consola
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Configurar token de Hugging Face para descargas estables
# Solo se configura si el token existe y no es el marcador de posición por defecto
if config.HF_TOKEN and "tu_token" not in config.HF_TOKEN:
    os.environ["HF_TOKEN"] = config.HF_TOKEN

st.set_page_config(page_title="Agente RAG Groq", layout="wide", page_icon="🤖")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)

@st.cache_resource
def get_vector_client():
    return chromadb.PersistentClient(path=config.VECTORDB_PATH)

@st.cache_resource
def get_groq_client():
    return Groq(api_key=config.GROQ_API_KEY)

embedding_model = get_embedding_model()
vector_client = get_vector_client()
groq_client = get_groq_client()

st.title("🤖 Agente RAG: Streamlit + Groq + Chroma")

with st.sidebar:
    st.header("📂 Gestión de Documentos")
    uploaded_files = st.file_uploader(
        "Sube tus archivos (PDF, TXT, DOCX)", 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Procesar e Indexar"):
        if uploaded_files:
            if not os.path.exists(config.DATA_PATH):
                os.makedirs(config.DATA_PATH)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(config.DATA_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Generando embeddings e indexando..."):
                result_msg = run_ingestion(embedding_model)
                st.success(result_msg)
        else:
            st.warning("Por favor, sube al menos un archivo.")

query = st.text_input("Haz una pregunta sobre el contenido de tus documentos:")

if query:
    if not os.path.exists(config.VECTORDB_PATH):
        st.error("La base de datos vectorial no existe. Por favor, indexa documentos primero.")
    else:
        with st.spinner("Consultando a la IA..."):
            response, sources = ask_rag(query, embedding_model, vector_client, groq_client)
            st.markdown("### 📝 Respuesta:")
            st.write(response)
            
            with st.expander("🔍 Ver fuentes relevantes"):
                for i, doc in enumerate(sources):
                    st.info(f"**Fuente {i+1} - {doc.metadata.get('source', 'Archivo desconocido')}:**\n\n{doc.page_content}")