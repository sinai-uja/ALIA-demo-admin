"""Módulo rag_tramites_eval — evaluación cuantitativa del RAG de trámites.

Tooling offline: catalog loader + scorers custom + runner MLflow GenAI. No es
importado por `backend.main` ni por ningún router de runtime; vive aislado y
se invoca desde `scripts/run_tramites_eval.py` y `scripts/bootstrap_eval_dataset.py`.

"""
