"""
Módulo rag_tramites — Tests del indexer.

Verifica que el indexer crea la colección y que contiene
al menos un documento por municipio.
"""

from backend.rag_tramites.indexer import (
    _construir_documentos,
    _crear_documentos_chunked,
    _extraer_auth,
    _extraer_url,
)

class TestExtractores:
    """Tests de las funciones auxiliares de extracción."""

    def test_extraer_url_con_url(self):
        texto = "Más info en https://sede.municipio.es/tramite y llamar."
        assert _extraer_url(texto) == "https://sede.municipio.es/tramite"

    def test_extraer_url_sin_url(self):
        assert _extraer_url("No hay enlace aquí") == ""

    def test_extraer_auth_certificado(self):
        assert _extraer_auth("Requiere certificado digital") == "certificado digital"

    def test_extraer_auth_clave(self):
        assert _extraer_auth("Acceso con Cl@ve") == "Cl@ve"

    def test_extraer_auth_sin_especificar(self):
        assert _extraer_auth("Trámite presencial") == "no especificado"

class TestConstruirDocumentos:
    """Tests de la construcción de documentos desde JSON."""

    def test_formato_dict(self):
        data = {
            "Adamuz": [
                {"nombre": "Empadronamiento", "texto": "Trámite de empadronamiento en Adamuz"}
            ]
        }
        docs = _construir_documentos(data)
        assert len(docs) == 1
        assert docs[0].metadata["municipio"] == "Adamuz"
        assert docs[0].metadata["nombre"] == "Empadronamiento"

    def test_formato_lista(self):
        data = [
            {"municipio": "Córdoba", "nombre": "Licencia obra", "texto": "Solicitud de licencia"}
        ]
        docs = _construir_documentos(data)
        assert len(docs) == 1
        assert docs[0].metadata["municipio"] == "Córdoba"

    def test_multiples_municipios(self):
        data = {
            "Adamuz": [{"nombre": "T1", "texto": "texto1"}],
            "Baena": [{"nombre": "T2", "texto": "texto2"}],
        }
        docs = _construir_documentos(data)
        municipios = {d.metadata["municipio"] for d in docs}
        assert "Adamuz" in municipios
        assert "Baena" in municipios

class TestChunkingPorSecciones:
    """Chunking del trámite por headers `## ` del markdown."""

    def test_tramite_simple_sin_headers(self):
        """Un trámite sin `## ` produce un único chunk de cabecera."""
        docs = _crear_documentos_chunked(
            "Adamuz", "Trámite simple", "Solo un párrafo de descripción."
        )
        assert len(docs) == 1
        assert docs[0].metadata["section"] == "Cabecera"
        assert docs[0].metadata["chunk_index"] == 0
        assert docs[0].metadata["total_chunks"] == 1
        assert docs[0].metadata["id"]  # slugified id presente

    def test_tramite_con_secciones(self):
        """Un trámite con headers `## ` produce un chunk por sección."""
        texto = (
            "PROCEDIMIENTO: Padrón\nEntidad: Adamuz\nID: 1234\n\n"
            "## Objeto\nInscripción en el padrón municipal.\n\n"
            "## Documentación\nDNI y justificante de empadronamiento.\n\n"
            "## Plazo\n10 días hábiles."
        )
        docs = _crear_documentos_chunked("Adamuz", "Padrón", texto)
        # 1 cabecera + 3 secciones
        assert len(docs) == 4
        sections = [d.metadata["section"] for d in docs]
        assert sections == ["Cabecera", "Objeto", "Documentación", "Plazo"]
        # Todos los chunks comparten tramite_id, municipio, nombre.
        ids = {d.metadata["id"] for d in docs}
        assert len(ids) == 1
        # chunk_index incremental y total_chunks consistente.
        indices = [d.metadata["chunk_index"] for d in docs]
        assert indices == [0, 1, 2, 3]
        for d in docs:
            assert d.metadata["total_chunks"] == 4

    def test_page_content_prefijo_nombre_seccion(self):
        """page_content prefija 'nombre — section. ...' para reforzar el embedding."""
        texto = "## Objeto\nDescripción del objeto del trámite."
        docs = _crear_documentos_chunked("Baena", "Padrón", texto)
        objeto_chunk = [d for d in docs if d.metadata["section"] == "Objeto"][0]
        assert objeto_chunk.page_content.startswith("Padrón — Objeto.")

    def test_seccion_muy_larga_se_subdivide(self):
        """Una sección >3500 chars se sub-split sin perder section."""
        cuerpo_largo = "Texto. " * 600  # ~4200 chars
        texto = f"## Documentación necesaria\n{cuerpo_largo}"
        docs = _crear_documentos_chunked("Cabra", "Tramitón", texto)
        secciones = [d for d in docs if d.metadata["section"] == "Documentación necesaria"]
        assert len(secciones) >= 2, "Sección larga debería partirse en >=2 chunks"
        # Cada chunk respeta el límite.
        for d in secciones:
            assert len(d.page_content) <= 4000  # margen para el prefijo

    def test_metadata_municipio_nombre_url_auth(self):
        texto = (
            "## Objeto\nSolicitud genérica.\n\n"
            "## URL\nhttps://example.com/tramite\n\n"
            "## Notas\nAcceso con Cl@ve."
        )
        docs = _crear_documentos_chunked("Lucena", "MiTramite", texto)
        for d in docs:
            assert d.metadata["municipio"] == "Lucena"
            assert d.metadata["nombre"] == "MiTramite"
            assert d.metadata["url"] == "https://example.com/tramite"
            assert d.metadata["auth"] == "Cl@ve"
