"""Caché in-memory con TTL para el PromptProvider.

Implementación minimalista basada en `dict` + `time.monotonic()`. Sin
dependencias externas. Suficiente para el tamaño y patrón de acceso del
proyecto (decenas de claves, lecturas frecuentes en hot path).

Thread-safety: el GIL hace atómicos los `dict.__getitem__` / `__setitem__`
individuales. La caché no garantiza atomicidad compuesta entre `get` y `set`
ni protección contra carreras: dos requests que pidan la misma clave en
miss simultáneo cargarán de MLflow en paralelo (aceptable; idempotente).

"""

from __future__ import annotations

import time
from typing import Any, Hashable, Optional

class TTLCache:
    """Caché clave→valor con expiración temporal absoluta.

    El TTL se fija en el constructor y aplica a todas las entradas. Cada
    `set` reinicia el reloj de la clave. Un `get` sobre una clave expirada
    la elimina y devuelve `None`.

    Si `ttl_seconds <= 0`, la caché no almacena nada: cada `get` devuelve
    `None`. Útil para tests que quieran desactivarla sin cambiar el código
    que la consume.
    """

    def __init__(self, *, ttl_seconds: int) -> None:
        self._ttl: float = max(0.0, float(ttl_seconds))
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        if self._ttl == 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            # Expirada — limpia y devuelve miss.
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if self._ttl == 0:
            # Modo "deshabilitado": no almacenes nada para no falsear el get.
            return
        self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        """Vacía la caché — útil para tests."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
