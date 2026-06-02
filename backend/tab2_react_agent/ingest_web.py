"""
Script de ingesta de documentos FAQ desde web scraping (Tab 2).

Lee URLs desde urls.txt, hace web scraping con Playwright,
extrae pares pregunta/respuesta y los persiste en ChromaDB.

Ejecución: python -m backend.tab2_react_agent.ingest_web
"""

import os
import re

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from playwright.sync_api import sync_playwright

from backend.config import settings

_urls_file = "./urls.txt"

def load_urls() -> list[str]:
    """Lee las URLs desde el fichero urls.txt, una por línea."""
    if not os.path.exists(_urls_file):
        print(f"El fichero '{_urls_file}' no existe. Créalo con una URL por línea.")
        return []

    urls = []
    with open(_urls_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    return urls

def scrape_page(url: str) -> str:
    """Obtiene el HTML renderizado de una URL usando Playwright."""
    print(f"Scraping: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
    return html

def extract_faq_chunks(html: str, source_url: str) -> list[Document]:
    """Extrae pares pregunta/respuesta del HTML de una página FAQ.

    Cada par se convierte en un Document de LangChain con metadata.
    Las preguntas se identifican por tags h4 con anchor links.
    """
    soup = BeautifulSoup(html, "html.parser")
    documents = []

    # Buscar todos los h4 que contienen preguntas FAQ
    headings = soup.find_all("h4")

    for heading in headings:
        question_text = heading.get_text(strip=True)

        # Limpiar prefijos de iconos/hashtags de gitbook
        question_text = re.sub(r"^(hashtag|#)+\s*", "", question_text).strip()

        # Filtrar headings que no son preguntas (ej: "Índice")
        if not question_text or len(question_text) < 10:
            continue

        # Recopilar contenido de respuesta: todo entre este h4 y el siguiente h4 o hr
        response_parts = []
        sibling = heading.find_next_sibling()
        while sibling:
            tag_name = sibling.name if sibling.name else ""
            if tag_name in ("h4", "h3", "h2", "h1"):
                break
            if tag_name == "hr":
                break
            text = sibling.get_text(strip=True)
            if text:
                response_parts.append(text)
            sibling = sibling.find_next_sibling()

        response_text = "\n".join(response_parts)

        if not response_text:
            continue

        # Construir el chunk semántico
        chunk_content = f"Pregunta: {question_text}\n\nRespuesta: {response_text}"

        # Si la respuesta es muy larga, subdividir manteniendo la pregunta como prefijo
        max_chunk_size = 2000
        if len(chunk_content) <= max_chunk_size:
            documents.append(Document(
                page_content=chunk_content,
                metadata={"source": source_url, "question": question_text},
            ))
        else:
            # Subdividir respuesta en fragmentos
            prefix = f"Pregunta: {question_text}\n\nRespuesta (continuación): "
            available = max_chunk_size - len(prefix)
            for i in range(0, len(response_text), available):
                fragment = response_text[i:i + available]
                if i == 0:
                    content = f"Pregunta: {question_text}\n\nRespuesta: {fragment}"
                else:
                    content = f"{prefix}{fragment}"
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source_url,
                        "question": question_text,
                        "part": i // available + 1,
                    },
                ))

    return documents

def ingest_web():
    """Proceso principal de ingesta web: scraping, chunking FAQ, embeddings y persistencia."""
    print("=" * 60)
    print("Ingesta web de documentos FAQ en ChromaDB")
    print("=" * 60)

    # Cargar URLs
    urls = load_urls()
    if not urls:
        print("No hay URLs para procesar.")
        return

    print(f"URLs a procesar: {len(urls)}")

    # Procesar cada URL
    all_documents = []
    for url in urls:
        try:
            html = scrape_page(url)
            chunks = extract_faq_chunks(html, source_url=url)
            print(f"  → Chunks extraídos: {len(chunks)}")
            all_documents.extend(chunks)
        except Exception as e:
            print(f"  → Error procesando {url}: {e}")

    if not all_documents:
        print("\nNo se extrajeron documentos de ninguna URL.")
        return

    print(f"\nTotal de chunks FAQ: {len(all_documents)}")

    # Mostrar preview de los chunks
    for i, doc in enumerate(all_documents, 1):
        question = doc.metadata.get("question", "")
        print(f"  [{i}] {question[:80]}...")

    # Modelo local de embeddings — no requiere API key externa
    print(f"\nGenerando embeddings con modelo: {settings.EMBEDDING_MODEL_PATH}")
    print(f"Dispositivo: {settings.EMBEDDING_DEVICE}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name="knowledge_base",
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    print(f"\nIngesta completada. Vectores almacenados en: {settings.CHROMA_PERSIST_DIR}")
    print("Colección: knowledge_base")
    print(f"Total de chunks indexados: {len(all_documents)}")
    print("=" * 60)

if __name__ == "__main__":
    ingest_web()
