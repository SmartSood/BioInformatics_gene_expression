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

# Install all common + ML dependencies once.
# --index-url points torch at PyTorch's CPU-only wheel index first - Tier 2
# runs on Graviton (t4g, no GPU), and PyPI's default `torch` wheel for
# linux-aarch64 is CUDA-enabled and pulls in ~2GB of nvidia-cudnn/cuda-toolkit
# packages that would never be used. --extra-index-url falls back to PyPI for
# everything else that isn't published on the PyTorch index.
RUN python -m pip install --upgrade pip && \
    python -m pip install --target=/install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
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
        torch \
        transformers \
        gensim \
        rdkit \
        unimol-tools

# xgboost installed separately with --no-deps: its PyPI wheel declares
# nvidia-nccl-cu13 as a hard dependency on *any* Linux platform (not gated
# on GPU presence - see pypi.org/pypi/xgboost/3.4.1/json), which is another
# multi-hundred-MB unused CUDA package on this GPU-less Graviton node.
# xgboost's only real runtime deps are numpy and scipy, both already
# installed above (numpy directly, scipy transitively via scikit-learn).
RUN python -m pip install --target=/install --no-deps xgboost

# Runtime stage
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy all pre-built dependencies
COPY --from=builder /install /usr/local/lib/python3.13/site-packages

# Create app directories
RUN mkdir -p /app/apps /app/packages
