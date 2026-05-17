FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/affinity_backend:/app

WORKDIR /app

# Copy only necessary app files
COPY apps/affinity_backend ./apps/affinity_backend
COPY packages ./packages

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "apps.affinity_backend.server:app", "--host", "0.0.0.0", "--port", "8003"]
