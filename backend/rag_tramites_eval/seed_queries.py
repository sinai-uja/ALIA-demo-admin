"""Seed queries para el dataset de evaluación del RAG de trámites.

Lista curada manualmente con la distribución mínima:

- 5 listados completos por municipio.
- 10 consultas específicas (un trámite concreto).
- 3 queries de scope mismatch (procedimientos no cubiertos por el catálogo).
- 6 edge cases (sin municipio, nombre en minúsculas, ambigua, etc.).

Esta lista es la **fuente de verdad reproducible** del dataset MLflow GenAI
`rag-tramites-eval-baseline-YYYY-MM-DD`. El bootstrap
(`scripts/bootstrap_eval_dataset.py`) la lee, expande el sentinel
`EXPECTED_ALL_TRAMITES` contra el catálogo real y registra cada entrada como
una traza anotada con `expected_response` y los campos custom.

`expected_tramite_ids`:
- Lista explícita de IDs (formato slugify del indexer).
- Sentinel `EXPECTED_ALL_TRAMITES` cuando se esperan todos los trámites del
  municipio (queries de tipo listado completo) — se expande en bootstrap.

`expected_scope`:
- `"in"` → la pregunta está cubierta por el catálogo; el LLM NO debe emitir
  disclaimer de scope mismatch.
- `"out"` → procedimiento no cubierto (solo hay productos derivados o nada);
  el LLM SÍ debe emitir el disclaimer.
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union

EXPECTED_ALL_TRAMITES = "__ALL__"

class SeedQuery(TypedDict):
    query: str
    expected_municipio: Union[str, None]
    expected_tramite_ids: Union[list[str], Literal["__ALL__"]]
    expected_response: str
    expected_scope: Literal["in", "out"]

SEED_QUERIES: list[SeedQuery] = [
    # ─────────────────────────────────────────────────────────────────────
    # LISTADOS (5) — el LLM debe enumerar TODOS los trámites del municipio.
    # ─────────────────────────────────────────────────────────────────────
    {
        "query": "¿Qué trámites están disponibles en Adamuz?",
        "expected_municipio": "Adamuz",
        "expected_tramite_ids": EXPECTED_ALL_TRAMITES,
        "expected_response": (
            "Adamuz ofrece 8 trámites: Actualización datos bancarios de terceros, "
            "Autoliquidación impuesto Plusvalías, Certificado de Empadronamiento "
            "Individual (Obtención automática), Derecho de Acceso a la Información, "
            "Pagos a proveedores: facturas, Publicación de edictos, Registro de "
            "entrada y Volante de Empadronamiento. Cada uno con su URL del portal "
            "e-admin de Eprinsa."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Listado completo de trámites del Ayuntamiento de Lucena",
        "expected_municipio": "Lucena",
        "expected_tramite_ids": EXPECTED_ALL_TRAMITES,
        "expected_response": (
            "Lucena tiene 19 trámites disponibles, incluyendo Ayuda al alquiler "
            "de vivienda para Jóvenes, Aplazamiento de pago de deudas tributarias, "
            "Multas de tráfico, Subvenciones a Emprendedores 'LUCENA EMPRENDE', "
            "y otros relacionados con padrón, escrituras, facturas y subvenciones."
        ),
        "expected_scope": "in",
    },
    {
        "query": "¿Cuáles son los trámites que ofrece Baena?",
        "expected_municipio": "Baena",
        "expected_tramite_ids": EXPECTED_ALL_TRAMITES,
        "expected_response": (
            "Baena dispone de trámites como Licencia de Obra Menor, Certificado "
            "de Empadronamiento Automatizado, Devolución Fianzas, Reserva de "
            "Instalaciones Deportivas, Licencia Tenencia Animales Peligrosos, "
            "Certificados de convivencia y otros. Listado completo con URLs en "
            "el contexto."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Trámites disponibles en Cabra",
        "expected_municipio": "Cabra",
        "expected_tramite_ids": EXPECTED_ALL_TRAMITES,
        "expected_response": (
            "Cabra cuenta con trámites de Certificados de Empadronamiento, "
            "Declaración IIVTNU-Plusvalía Municipal, Inscripción de parejas de "
            "hecho, Devolución de garantías, Registro de Asociaciones, Tramitación "
            "de bodas civiles y otros, listados con sus URLs respectivas."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Enumera todos los trámites de Aguilar",
        "expected_municipio": "Aguilar",
        "expected_tramite_ids": EXPECTED_ALL_TRAMITES,
        "expected_response": (
            "Aguilar tiene los trámites siguientes en su sede e-admin, "
            "incluyendo Pagos a proveedores, Registro de entrada y Reclamaciones "
            "y aportación de documentación a procesos selectivos."
        ),
        "expected_scope": "in",
    },
    # ─────────────────────────────────────────────────────────────────────
    # ESPECÍFICAS (10) — el LLM debe devolver UN trámite concreto.
    # ─────────────────────────────────────────────────────────────────────
    {
        "query": "¿Cómo obtengo el volante de empadronamiento en Adamuz?",
        "expected_municipio": "Adamuz",
        "expected_tramite_ids": ["adamuz-volante-de-empadronamiento"],
        "expected_response": (
            "El trámite 'Volante de Empadronamiento' de Adamuz está disponible "
            "en https://e-admin.eprinsa.es/adamuz/... con acceso mediante Cl@ve."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Plusvalía municipal en Adamuz",
        "expected_municipio": "Adamuz",
        "expected_tramite_ids": ["adamuz-autoliquidacion-impuesto-plusvalias"],
        "expected_response": (
            "Adamuz ofrece la 'Autoliquidación impuesto Plusvalías' para "
            "transmisiones de inmuebles urbanos. URL del portal Eprinsa."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Quiero obtener un certificado de empadronamiento en Baena",
        "expected_municipio": "Baena",
        "expected_tramite_ids": ["baena-certificado-de-empadronamiento-automatizado"],
        "expected_response": (
            "Baena dispone del 'Certificado de Empadronamiento Automatizado'. "
            "Se accede desde la sede e-admin de Baena."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Licencia de obra menor en Baena",
        "expected_municipio": "Baena",
        "expected_tramite_ids": ["baena-licencia-de-obra-menor"],
        "expected_response": (
            "El trámite 'Licencia de Obra Menor' de Baena permite solicitar "
            "obras menores en el municipio. URL en la sede electrónica."
        ),
        "expected_scope": "in",
    },
    {
        "query": "¿Cómo registro a mi pareja de hecho en Cabra?",
        "expected_municipio": "Cabra",
        "expected_tramite_ids": ["cabra-inscripcion-de-parejas-de-hecho"],
        "expected_response": (
            "Cabra ofrece el trámite 'Inscripción de parejas de hecho' en el "
            "portal e-admin del municipio."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Multas de tráfico en Lucena",
        "expected_municipio": "Lucena",
        "expected_tramite_ids": ["lucena-gestion-de-multas-de-trafico"],
        "expected_response": (
            "Lucena dispone del trámite 'Gestión de Multas de Tráfico' para "
            "consultar y pagar sanciones. URL del portal municipal."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Ayuda al alquiler de vivienda para jóvenes en Lucena",
        "expected_municipio": "Lucena",
        "expected_tramite_ids": ["lucena-ayuda-al-alquiler-de-vivienda-para-jovenes"],
        "expected_response": (
            "Lucena ofrece la 'Ayuda al alquiler de vivienda para Jóvenes' "
            "dentro de sus convocatorias de subvenciones."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Aplazamiento o fraccionamiento de pago de deuda tributaria en Lucena",
        "expected_municipio": "Lucena",
        "expected_tramite_ids": [
            "lucena-aplazamiento-o-fraccionamiento-de-pago-de-deudas-tributar"
        ],
        "expected_response": (
            "Lucena permite solicitar 'Aplazamiento o fraccionamiento de pago "
            "de deudas tributarias en periodo voluntario'."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Reserva de instalaciones deportivas en Baena",
        "expected_municipio": "Baena",
        "expected_tramite_ids": ["baena-reserva-de-instalaciones-deportivas"],
        "expected_response": (
            "Baena ofrece el trámite 'Reserva de Instalaciones Deportivas' "
            "para reservar pistas y polideportivos municipales."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Licencia para tener un perro potencialmente peligroso en Baena",
        "expected_municipio": "Baena",
        "expected_tramite_ids": ["baena-licencia-tenencia-animales-peligrosos"],
        "expected_response": (
            "Baena tramita la 'Licencia Tenencia Animales Peligrosos' para "
            "perros PPP y otros animales considerados peligrosos."
        ),
        "expected_scope": "in",
    },
    {
        "query": "Devolución de fianza de obra en Baena",
        "expected_municipio": "Baena",
        "expected_tramite_ids": ["baena-devolucion-fianzas"],
        "expected_response": (
            "Baena dispone del trámite 'Devolución Fianzas' para recuperar "
            "fianzas depositadas en obras y otros procedimientos."
        ),
        "expected_scope": "in",
    },
    # ─────────────────────────────────────────────────────────────────────
    # SCOPE MISMATCH (3) — procedimientos no cubiertos.
    # ─────────────────────────────────────────────────────────────────────
    {
        # Adamuz solo tiene Volante y Certificado (productos derivados), no
        # el alta padronal en sí. La regla scope-mismatch debe disparar.
        "query": "¿Cómo me empadrono en Adamuz?",
        "expected_municipio": "Adamuz",
        "expected_tramite_ids": [
            "adamuz-volante-de-empadronamiento",
            "adamuz-certificado-de-empadronamiento-individual-obtencion-autom",
        ],
        "expected_response": (
            "Aviso: el catálogo de Adamuz no incluye el procedimiento completo "
            "de alta en el Padrón Municipal. Debes acudir al Ayuntamiento o a "
            "la sede electrónica oficial. Como información complementaria, hay "
            "trámites para obtener volantes y certificados de empadronamiento."
        ),
        "expected_scope": "out",
    },
    {
        # No hay trámite de licencia de actividad / apertura de negocio en Lucena.
        "query": "¿Cómo abro un negocio en Lucena?",
        "expected_municipio": "Lucena",
        "expected_tramite_ids": [],
        "expected_response": (
            "Aviso: el catálogo de trámites de Lucena no incluye un "
            "procedimiento específico de apertura de negocio o licencia de "
            "actividad. Debes consultar directamente con el Ayuntamiento o la "
            "sede electrónica oficial del municipio."
        ),
        "expected_scope": "out",
    },
    {
        # No hay trámite de adjudicación de vivienda social en Adamuz.
        "query": "¿Cómo solicito una vivienda social en Adamuz?",
        "expected_municipio": "Adamuz",
        "expected_tramite_ids": [],
        "expected_response": (
            "Aviso: el catálogo de Adamuz no incluye un trámite específico "
            "para solicitar vivienda social. Acude al Ayuntamiento o a la "
            "sede electrónica para consultar el procedimiento."
        ),
        "expected_scope": "out",
    },
    # ─────────────────────────────────────────────────────────────────────
    # EDGE CASES (6) — sin municipio, multi-municipio, scope-in con
    # procedimiento amplio cubierto, nombre en minúsculas.
    # ─────────────────────────────────────────────────────────────────────
    {
        # Cabra SÍ tiene 'Tramitación bodas civiles', por tanto scope_in
        # — el disclaimer sería falso positivo aquí.
        "query": "¿Cómo me caso en Cabra?",
        "expected_municipio": "Cabra",
        "expected_tramite_ids": ["cabra-tramitacion-bodas-civiles"],
        "expected_response": (
            "Cabra ofrece el trámite 'Tramitación bodas civiles' para la "
            "celebración de matrimonios civiles en el municipio."
        ),
        "expected_scope": "in",
    },
    {
        # Query sin municipio — el nodo debe devolver None.
        "query": "¿Dónde puedo solicitar un certificado de empadronamiento?",
        "expected_municipio": None,
        "expected_tramite_ids": [],
        "expected_response": (
            "El trámite de certificado de empadronamiento está disponible en "
            "varios municipios del catálogo. Indica de qué municipio te "
            "interesa para darte la URL concreta."
        ),
        "expected_scope": "in",
    },
    {
        # Municipio en minúsculas — debe normalizar y detectar 'Lucena'.
        "query": "trámites tributarios en lucena",
        "expected_municipio": "Lucena",
        "expected_tramite_ids": [
            "lucena-aplazamiento-o-fraccionamiento-de-pago-de-deudas-tributar",
            "lucena-gestion-de-multas-de-trafico",
            "lucena-presentacion-de-facturas-no-obligados-a-face",
            "lucena-presentacion-de-facturas-electronicas-face",
        ],
        "expected_response": (
            "Lucena dispone de trámites tributarios: Aplazamiento o "
            "fraccionamiento de pago de deudas, Gestión de Multas de Tráfico, "
            "y Presentación de facturas (FACe y no FACe)."
        ),
        "expected_scope": "in",
    },
    {
        # Query muy ambigua sin municipio — el LLM debe pedir aclaración o
        # listar genéricamente sin alucinar URLs concretas.
        "query": "Quiero hacer un registro de entrada",
        "expected_municipio": None,
        "expected_tramite_ids": [],
        "expected_response": (
            "El trámite 'Registro de entrada' está disponible en la mayoría "
            "de municipios del catálogo. Indica el municipio para darte la URL."
        ),
        "expected_scope": "in",
    },
    {
        # Adamuz en minúsculas — detección case-insensitive.
        "query": "Quiero saber sobre la plusvalía en adamuz",
        "expected_municipio": "Adamuz",
        "expected_tramite_ids": ["adamuz-autoliquidacion-impuesto-plusvalias"],
        "expected_response": (
            "Adamuz ofrece la 'Autoliquidación impuesto Plusvalías'. Esta es "
            "la URL del trámite en el portal Eprinsa."
        ),
        "expected_scope": "in",
    },
    {
        # Municipio inexistente en el catálogo — el nodo debe devolver None
        # y el LLM no debe alucinar URLs.
        "query": "¿Qué trámites hay en Madrid?",
        "expected_municipio": None,
        "expected_tramite_ids": [],
        "expected_response": (
            "Madrid no está incluido en el catálogo de trámites de la "
            "Diputación de Córdoba. Consulta la sede electrónica del "
            "Ayuntamiento de Madrid directamente."
        ),
        "expected_scope": "in",
    },
]
