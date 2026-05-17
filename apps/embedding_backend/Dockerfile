FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/embedding_backend:/app \
    HF_HOME=/tmp/hf_cache

WORKDIR /app

# Copy only necessary app files (exclude hf_cache)
COPY apps/embedding_backend ./apps/embedding_backend
COPY packages ./packages

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "apps.embedding_backend.server:app", "--host", "0.0.0.0", "--port", "8002"]
