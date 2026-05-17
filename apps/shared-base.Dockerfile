# Shared base image for all services
# Installs common dependencies + heavy ML libraries (torch, transformers, etc.)
# This layer is built once and reused across multiple services

FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ git libgomp1 libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /install
WORKDIR /install

# Install all common + ML dependencies once
RUN python -m pip install --upgrade pip && \
    python -m pip install --target=/install \
        fastapi \
        "uvicorn[standard]" \
        redis \
        rq \
        python-dotenv \
        pandas \
        numpy \
        joblib \
        scikit-learn \
        prisma \
        boto3 \
        PyJWT \
        pydantic \
        mlflow \
        xgboost \
        torch \
        transformers \
        gensim \
        rdkit \
        unimol-tools

# Runtime stage
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy all pre-built dependencies
COPY --from=builder /install /usr/local/lib/python3.13/site-packages

# Create app directories
RUN mkdir -p /app/apps /app/packages
