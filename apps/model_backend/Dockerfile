FROM smarthcnbl/shared-base:latest

ENV PYTHONPATH=/app/apps/model_backend:/app

WORKDIR /app

# Copy only necessary app files
COPY apps/model_backend ./apps/model_backend
COPY packages ./packages

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "apps.model_backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
