# Stage 1: Builder — instalar dependencias
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Production — imagen limpia
FROM python:3.11-slim

# Copiar entorno virtual del builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar curl para healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m appuser

WORKDIR /app

# Copiar código fuente
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY urls.txt ./
COPY requirements.txt ./

# Crear directorios necesarios con permisos
RUN mkdir -p chroma_db logs docs && \
    chown -R appuser:appuser /app

USER appuser

# Puerto del backend (por defecto)
EXPOSE 8000

# Healthcheck contra el endpoint /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Arrancar backend con uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
