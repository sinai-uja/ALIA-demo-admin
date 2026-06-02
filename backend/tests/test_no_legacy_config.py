"""Test guardia: ningún módulo del backend usa lecturas legacy de variables de entorno.

Tras la  (Unificar configuración entre módulos), todo el backend lee la
configuración a través de `from backend.config import settings`. Este test
recorre los archivos del backend (excluyendo tests) y falla si encuentra
`os.getenv(...)`, `os.environ[...]`, `os.environ.get(...)` o `load_dotenv(...)`.

"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

# Patrones prohibidos: llamadas reales, no menciones en strings.
_PATTERNS = [
    re.compile(r"\bload_dotenv\s*\("),
    re.compile(r"\bos\.getenv\s*\("),
    re.compile(r"\bos\.environ\s*\["),
    re.compile(r"\bos\.environ\.get\s*\("),
    re.compile(r"^\s*from\s+dotenv\s+import"),
    re.compile(r"^\s*import\s+dotenv\b"),
]

# `backend/config.py` menciona los patrones en su docstring como referencia
# histórica al patrón que sustituye. Es la única exención legítima.
_EXEMPT = {
    BACKEND / "config.py",
}

def _backend_python_files():
    for path in BACKEND.rglob("*.py"):
        parts = path.parts
        if "__pycache__" in parts:
            continue
        if "tests" in parts:
            continue
        if path in _EXEMPT:
            continue
        yield path

def test_no_legacy_env_reads_in_backend():
    offenders = []
    for path in _backend_python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in _PATTERNS:
                if pattern.search(line):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}"
                    )
                    break

    assert not offenders, (
        "Encontrada lectura legacy de variables de entorno en backend. "
        "Usa `from backend.config import settings` en su lugar.\n"
        + "\n".join(offenders)
    )
