FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/affinity_backend:/app

WORKDIR /app

# Copy only necessary app files
COPY apps/affinity_backend ./apps/affinity_backend
# affinity_backend imports its base auth deps and S3 helpers from
# model_backend (apps.model_backend.auth, apps.model_backend.storage)
# rather than a shared package - only pull in those self-contained modules,
# not the whole model_backend app.
COPY apps/model_backend/auth ./apps/model_backend/auth
COPY apps/model_backend/storage ./apps/model_backend/storage
COPY packages ./packages

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "apps.affinity_backend.server:app", "--host", "0.0.0.0", "--port", "8003"]
