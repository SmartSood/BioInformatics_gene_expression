FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/depmap_backend:/app

WORKDIR /app

# Copy only necessary app files
COPY apps/depmap_backend ./apps/depmap_backend
# depmap_backend imports its base auth deps and S3 helpers from
# model_backend (apps.model_backend.auth, apps.model_backend.storage)
# rather than a shared package - only pull in those self-contained modules,
# not the whole model_backend app.
COPY apps/model_backend/auth ./apps/model_backend/auth
COPY apps/model_backend/storage ./apps/model_backend/storage
# associations.py/depmap_worker.py do `from client.db import ...` (bare
# top-level import, not apps.model_backend.client) - copied into
# apps/depmap_backend/client so it resolves via this service's own
# PYTHONPATH entry (/app/apps/depmap_backend), same as model_backend
# resolves its own copy of this module via its own PYTHONPATH.
COPY apps/model_backend/client ./apps/depmap_backend/client
# depmap_worker.py inserts ../depmap onto sys.path at runtime and imports
# depmap_associations.py plus its ~1.2GB bundled public reference dataset
# (expression/CTRP/GDSC/PRISM matrices) directly - not S3-backed, so it has
# to physically ship in this image.
COPY apps/depmap ./apps/depmap
COPY packages ./packages

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "apps.depmap_backend.server:app", "--host", "0.0.0.0", "--port", "8001"]
