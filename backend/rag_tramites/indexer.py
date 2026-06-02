"""
Módulo rag_tramites — Script offline de indexación de trámites en ChromaDB.

Lee el fichero JSON de trámites (export.json), extrae campos relevantes,
genera embeddings con HuggingFaceEmbeddings (MrBERT) y persiste en una
colección dedicada de ChromaDB.

Ejecución: python -m backend.rag_tramites.indexer
"""

import json
import logging
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from slugify import slugify

from backend.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Funciones auxiliares de extracción ─────────────────────────

def _extraer_url(texto: str) -> str:
    """Extrae la primera URL del texto."""
    match = re.search(r"https?://\S+", texto)
    return match.group(0).rstrip(".,;)") if match else ""

def _extraer_auth(texto: str) -> str:
    """Extrae el tipo de autenticación requerida del texto."""
    texto_lower = texto.lower()
    if "certificado digital" in texto_lower or "certificado electrónico" in texto_lower:
        return "certificado digital"
    if "cl@ve" in texto_lower or "clave" in texto_lower:
        return "Cl@ve"
    if "sin identificación" in texto_lower or "no requiere" in texto_lower:
        return "sin autenticación"
    return "no especificado"

def _extraer_campo(texto: str, etiqueta: str) -> str:
    """Extrae el valor de un campo etiquetado en el texto (ej: 'Destinatarios: ...')."""
    pattern = rf"{re.escape(etiqueta)}\s*[:：]\s*(.+?)(?:\n|$)"
    match = re.search(pattern, texto, re.IGNORECASE)
    return match.group(1).strip() if match else ""

# ── Proceso de indexación ──────────────────────────────────────

def _cargar_json() -> dict:
    """Carga el fichero export.json."""
    logger.info(f"Cargando JSON desde: {settings.TRAMITES_DATA_PATH}")
    with open(settings.TRAMITES_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _construir_documentos(data: dict) -> list[Document]:
    """Convierte la estructura municipio→trámites en Documents de LangChain.

    Cada trámite puede generar varios Documents (chunks por sección)
    — ver `_crear_documentos_chunked`.

    Soporta dos formatos de JSON:
    - Dict de municipio → lista/dict de trámites
    - Lista plana de objetos con campo 'municipio'
    """
    documents = []

    if isinstance(data, list):
        for item in data:
            municipio = item.get("municipio", "desconocido")
            nombre = item.get("nombre", item.get("titulo", "sin nombre"))
            texto = item.get("texto", item.get("descripcion", json.dumps(item, ensure_ascii=False)))
            documents.extend(_crear_documentos_chunked(municipio, nombre, texto))
    elif isinstance(data, dict):
        for municipio, tramites in data.items():
            if isinstance(tramites, list):
                for tramite in tramites:
                    if isinstance(tramite, dict):
                        nombre = tramite.get("nombre", tramite.get("titulo", "sin nombre"))
                        texto = tramite.get("texto", tramite.get("descripcion",
                                            json.dumps(tramite, ensure_ascii=False)))
                    else:
                        nombre = str(tramite)
                        texto = str(tramite)
                    documents.extend(_crear_documentos_chunked(municipio, nombre, texto))
            elif isinstance(tramites, dict):
                for nombre, texto in tramites.items():
                    if isinstance(texto, dict):
                        texto = json.dumps(texto, ensure_ascii=False)
                    documents.extend(_crear_documentos_chunked(municipio, nombre, str(texto)))

    return documents

# Chunking por secciones del trámite. Cada sección markdown del trámite
# ("## Objeto", "## Plazo", "## Documentación", etc.) se embede por
# separado, de forma que una query dirigida a una sección concreta puede
# recuperarla aunque esté en la mitad/final del trámite.
#
# - Headers reconocidos: `##` (los trámites del catálogo usan exactamente
#   este nivel — el 100% de los trámites largos tiene al menos un `## `).
# - Si una sección supera `_SECCION_MAX_CHARS` (raro), sub-split con
#   `RecursiveCharacterTextSplitter` para no exceder los límites del
#   modelo de embeddings en aceleradores con tensor int32 (Apple MPS).
# - El texto del preámbulo del trámite (antes del primer `##` — típicamente
#   "PROCEDIMIENTO: ..., Entidad: ..., ID: ...") se preserva como una
#   sección "Cabecera" para no perderlo.
# - Trámites sin headers `##` se indexan como un único chunk con
#   section="Completo" (truncado a 4000 chars como fallback de seguridad).
_SECCION_MAX_CHARS = 3500
_TRAMITE_FLAT_MAX_CHARS = 4000

_markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "section")],
    strip_headers=True,
)
_seccion_subsplitter = RecursiveCharacterTextSplitter(
    chunk_size=_SECCION_MAX_CHARS,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def _crear_documentos_chunked(municipio: str, nombre: str, texto: str) -> list[Document]:
    """Convierte un trámite en uno o más Documents, chunkeando por secciones."""
    tramite_id = slugify(f"{municipio}_{nombre}")[:64]
    base_metadata = {
        "municipio": municipio,
        "nombre": nombre,
        "url": _extraer_url(texto),
        "auth": _extraer_auth(texto),
        "delegacion": _extraer_campo(texto, "Delegación"),
        "tematica": _extraer_campo(texto, "Temática"),
        "destinatarios": _extraer_campo(texto, "Destinatarios"),
        "id": tramite_id,
    }

    # Preámbulo (texto antes del primer `## `) — lo preservamos como sección
    # "Cabecera" para no perder los campos estructurados (Procedimiento,
    # Entidad, ID, Delegación) que LangChain descartaría con strip_headers.
    # `has_headers` cubre tanto el caso normal ("preámbulo\n## ...") como
    # el caso degenerado donde el texto empieza directamente con `## `.
    has_headers = "\n## " in texto or texto.startswith("## ")
    if "\n## " in texto:
        preamble, _ = texto.split("\n## ", 1)
    elif texto.startswith("## "):
        preamble = ""
    else:
        preamble = texto

    chunks: list[tuple[str, str]] = []  # (section, contenido)

    preamble = preamble.strip()
    if preamble:
        chunks.append(("Cabecera", preamble))

    if has_headers:
        # Split markdown SOLO desde el primer `## ` para que el splitter no
        # devuelva además el preámbulo como un chunk sin sección (ya lo
        # capturamos arriba con explícitamente section="Cabecera").
        idx_first_header = texto.index("## ")
        texto_headers = texto[idx_first_header:]
        split_docs = _markdown_splitter.split_text(texto_headers)
        for split_doc in split_docs:
            section = split_doc.metadata.get("section", "").strip()
            content = split_doc.page_content.strip()
            if not content or not section:
                # Sin section → es el bloque previo al primer header, ya
                # cubierto por el preámbulo. Descartar.
                continue
            if len(content) > _SECCION_MAX_CHARS:
                # Sección demasiado larga: sub-split.
                sub_chunks = _seccion_subsplitter.split_text(content)
                for sub in sub_chunks:
                    chunks.append((section, sub.strip()))
            else:
                chunks.append((section, content))
    elif not preamble:
        # Caso degenerado (sin preámbulo y sin headers): tratar texto
        # completo como un único chunk truncado.
        chunks.append(("Completo", texto[:_TRAMITE_FLAT_MAX_CHARS]))

    if not chunks:
        # Fallback defensivo: el trámite estaba vacío.
        return []

    total_chunks = len(chunks)
    documents = []
    for idx, (section, content) in enumerate(chunks):
        # Prefijo en page_content para que el embedding capture nombre +
        # sección como parte del vector semántico — mejora retrieval para
        # queries del tipo "documentación para X en Y".
        page_content = f"{nombre} — {section}. {content}"
        metadata = {
            **base_metadata,
            "section": section,
            "chunk_index": idx,
            "total_chunks": total_chunks,
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents

# Compatibilidad backwards para código y tests que esperaban un único
# Document por trámite. Devuelve el primer chunk (Cabecera o equivalente).
def _crear_documento(municipio: str, nombre: str, texto: str) -> Document:
    docs = _crear_documentos_chunked(municipio, nombre, texto)
    if not docs:
        raise ValueError(f"Trámite vacío: {municipio}/{nombre}")
    return docs[0]

def indexar():
    """Proceso principal de indexación: carga, transforma, embede y persiste."""
    logger.info("=" * 60)
    logger.info("Indexación de trámites municipales en ChromaDB")
    logger.info(f"Colección: {settings.TRAMITES_CHROMA_COLLECTION}")
    logger.info("=" * 60)

    # Cargar JSON
    data = _cargar_json()

    # Construir documentos
    documents = _construir_documentos(data)
    logger.info(f"Documentos construidos: {len(documents)}")

    if not documents:
        logger.warning("No se encontraron trámites para indexar.")
        return

    # Preview
    municipios = set(d.metadata["municipio"] for d in documents)
    logger.info(f"Municipios encontrados: {len(municipios)}")
    for i, doc in enumerate(documents[:3], 1):
        logger.info(f"  [{i}] {doc.metadata['municipio']} — {doc.metadata['nombre'][:60]}")
    if len(documents) > 3:
        logger.info(f"  ... y {len(documents) - 3} más")

    # Embeddings — mismo modelo que el resto del proyecto (MrBERT)
    logger.info(f"Modelo de embeddings: {settings.EMBEDDING_MODEL_PATH}")
    logger.info(f"Dispositivo: {settings.EMBEDDING_DEVICE}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Borrar colección existente para idempotencia
    try:
        existing = Chroma(
            collection_name=settings.TRAMITES_CHROMA_COLLECTION,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
        existing.delete_collection()
        logger.info(f"Colección '{settings.TRAMITES_CHROMA_COLLECTION}' eliminada (idempotencia)")
    except Exception:
        logger.info("No existía colección previa, se creará nueva")

    # Indexar en batches de 50
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=settings.TRAMITES_CHROMA_COLLECTION,
                persist_directory=settings.CHROMA_PERSIST_DIR,
            )
        else:
            vectorstore.add_documents(batch)
        logger.info(f"  Batch {i // batch_size + 1}: {len(batch)} docs indexados")

    logger.info("=" * 60)
    logger.info(f"Indexación completada: {len(documents)} trámites en '{settings.TRAMITES_CHROMA_COLLECTION}'")
    logger.info(f"ChromaDB: {settings.CHROMA_PERSIST_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    indexar()
