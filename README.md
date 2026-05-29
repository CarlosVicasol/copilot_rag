README - Agente RAG con Streamlit + Groq + Chroma

Objetivo

Construir un agente RAG (Retrieval-Augmented Generation) en Streamlit que:

- Permita subir documentos desde la interfaz web
- Indexe automáticamente esos documentos
- Responda preguntas sobre su contenido
- Use Groq API para generación
- Use embeddings locales para búsqueda semántica
- Use ChromaDB como base de datos vectorial

Arquitectura

Usuario (Streamlit)
    |
    v
Pregunta
    |
    v
Embeddings (modelo local)
    |
    v
ChromaDB (vector store)
    |
    v
Top-K fragmentos relevantes
    |
    v
Prompt RAG
    |
    v
Groq API
    |
    v
Respuesta

Estructura del proyecto

rag-agent/
|
|-- app.py
|-- ingest.py
|-- rag.py
|-- config.py
|
|-- data/
|-- vectordb/
|
|-- requirements.txt
|-- README.md

Requisitos

- Python 3.10 o superior

Crear entorno virtual

python -m venv venv

Linux/Mac:
source venv/bin/activate

Windows:
venv\Scripts\activate

Instalación de dependencias

pip install streamlit langchain chromadb sentence-transformers groq pypdf python-docx

Configuración API Groq

Linux/Mac:
export GROQ_API_KEY="tu_api_key"

Windows:
setx GROQ_API_KEY "tu_api_key"

Tipos de documentos soportados

- PDF
- TXT
- DOCX
- MD

Embeddings

Modelo recomendado:

sentence-transformers/all-MiniLM-L6-v2

Proceso de ingestión

Responsabilidades de ingest.py:

1. Leer documentos desde /data
2. Dividir en fragmentos
3. Generar embeddings
4. Guardar en ChromaDB

Ejemplo de fragmentación

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

Configuración Chroma

from langchain.vectorstores import Chroma

db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

Integración con Groq

from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {"role": "system", "content": "Eres un asistente experto en documentos"},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

Lógica RAG

Flujo:

1. Recibir pregunta
2. Convertir a embedding
3. Consultar ChromaDB
4. Recuperar documentos relevantes
5. Construir contexto
6. Llamar a Groq
7. Devolver respuesta

Prompt recomendado

Responde únicamente basándote en el contexto proporcionado.
Si no encuentras la respuesta, indica que no está en los documentos.

Contexto:
{context}

Pregunta:
{question}

Respuesta:

Interfaz Streamlit

Funciones:

- Subida de documentos
- Guardado en /data
- Ejecución de ingestión
- Entrada de preguntas
- Mostrar respuestas

Ejemplo básico

import streamlit as st

st.title("Agente RAG")

query = st.text_input("Haz una pregunta")

if query:
    respuesta = ask_rag(query)
    st.write(respuesta)

Workflow

1. Añadir documentos a /data
2. Ejecutar ingestión:
   python ingest.py
3. Ejecutar aplicación:
   streamlit run app.py

Parámetros recomendados

Chunk size: 400-600
Top-K: 3 a 5

Buenas prácticas

Persistencia:
persist_directory="vectordb"

Uso de metadata:

metadata = {
  "source": file_name,
  "tipo": "documento",
  "fecha": "2026-01-01"
}

Mostrar fuentes en respuesta

Control de errores:
- Ficheros vacíos
- Tipos no soportados
- Errores de API

Seguridad:
- Limitar tamaño de subida
- Validar formatos
- Sanitizar entradas

Mejoras futuras

- Re-ranking
- Multi-query RAG
- Streaming de respuestas
- Cache de embeddings
- Filtros por metadata
- Integración con sistemas externos

Checklist

- Entorno virtual creado
- Dependencias instaladas
- API Groq configurada
- Documentos cargados
- Indexación realizada
- Aplicación funcionando

Notas

Arquitectura basada en embeddings locales + Groq permite velocidad alta, coste bajo y despliegue sencillo.
