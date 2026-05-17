FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/depmap_backend:/app

WORKDIR /app

# Copy only necessary app files
COPY apps/depmap_backend ./apps/depmap_backend
COPY packages ./packages

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "apps.depmap_backend.server:app", "--host", "0.0.0.0", "--port", "8001"]
