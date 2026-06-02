"""
Script de ingesta de documentos para ChromaDB (Tab 2).

Lee documentos de la carpeta ./docs/ (acepta .txt, .pdf, .md),
hace chunking y genera embeddings para persistirlos en ChromaDB.

Usa un modelo de embeddings local (HuggingFace) definido en EMBEDDING_MODEL_PATH.

Ejecución: python -m backend.tab2_react_agent.ingest
"""

import os
import glob

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

_docs_dir = "./docs"

def load_documents():
    """Carga todos los documentos soportados desde la carpeta ./docs/."""
    documents = []

    # Mapeo de extensiones a loaders
    loaders_map = {
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader,
        "*.md": UnstructuredMarkdownLoader,
    }

    for pattern, loader_class in loaders_map.items():
        files = glob.glob(os.path.join(_docs_dir, pattern))
        for file_path in files:
            try:
                print(f"Cargando: {file_path}")
                loader = loader_class(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")

    return documents

def ingest():
    """Proceso principal de ingesta: carga, chunking, embeddings y persistencia."""
    print("=" * 60)
    print("Iniciando ingesta de documentos en ChromaDB")
    print("=" * 60)

    # Verificar que existe la carpeta de documentos
    if not os.path.exists(_docs_dir):
        print(f"La carpeta '{_docs_dir}' no existe. Créala y añade documentos.")
        return

    # Cargar documentos
    documents = load_documents()
    if not documents:
        print("No se encontraron documentos para ingestar.")
        return

    print(f"\nDocumentos cargados: {len(documents)}")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Chunks generados: {len(chunks)}")

    # Modelo local de embeddings — no requiere API key externa
    # CRÍTICO: debe ser idéntico al usado en rag_tools.py para compatibilidad vectorial
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Persistencia en ChromaDB (side-effect: escribe a CHROMA_PERSIST_DIR)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="knowledge_base",
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    print(f"\nIngesta completada. Vectores almacenados en: {settings.CHROMA_PERSIST_DIR}")
    print("Colección: knowledge_base")
    print(f"Modelo de embeddings: {settings.EMBEDDING_MODEL_PATH}")
    print(f"Dispositivo: {settings.EMBEDDING_DEVICE}")
    print(f"Total de chunks indexados: {len(chunks)}")
    print("=" * 60)

if __name__ == "__main__":
    ingest()
