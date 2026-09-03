FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/embedding_backend:/app \
    HF_HOME=/tmp/hf_cache

WORKDIR /app

# Copy only necessary app files (exclude hf_cache)
COPY apps/embedding_backend ./apps/embedding_backend
# embedding_backend imports its base auth deps and S3 helpers from
# model_backend (apps.model_backend.auth, apps.model_backend.storage)
# rather than a shared package - only pull in those self-contained modules,
# not the whole model_backend app.
COPY apps/model_backend/auth ./apps/model_backend/auth
COPY apps/model_backend/storage ./apps/model_backend/storage
# embedding_service.py inserts ../embedding_bundle/src/Drug_Embeddings/*
# onto sys.path at runtime for vendored Mol2Vec/GIN code (bare `from
# features import ...`), and appends embedding_bundle/models to sys.path
# for ~488MB of bundled pretrained model weights - not S3-backed, so both
# have to physically ship in this image.
COPY apps/embedding_bundle ./apps/embedding_bundle
COPY packages ./packages

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "apps.embedding_backend.server:app", "--host", "0.0.0.0", "--port", "8002"]
