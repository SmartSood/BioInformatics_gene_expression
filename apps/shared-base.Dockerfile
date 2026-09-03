# Shared base image for all services
# Installs common dependencies + heavy ML libraries (torch, transformers, etc.)
# This layer is built once and reused across multiple services

FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libatomic1: required only by the Node runtime that prisma-client-py's CLI
# bundles internally to run codegen (`prisma generate` below), a build-time
# step - not needed once the client is generated.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ git libgomp1 libssl-dev libffi-dev libatomic1 \
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

# python-multipart: required by FastAPI for any endpoint that accepts
# UploadFile/Form data (model_backend, embedding_backend, affinity_backend
# all have file-upload routes) - not pulled in automatically by fastapi
# itself.
RUN python -m pip install --target=/install python-multipart

# s3fs: lets pandas/fsspec read/write s3:// URIs directly (e.g.
# pd.read_csv("s3://...")) - train_worker.py does this for dataset files
# already uploaded to S3. Without it, fsspec raises "Install s3fs to
# access S3" the first time a job actually reads one.
RUN python -m pip install --target=/install s3fs

# Generate the Prisma Python client once here (not per-service): every
# backend that touches the DB does `from prisma import Prisma`, which fails
# at runtime with "Client hasn't been generated yet" unless `prisma generate`
# has run against packages/db/prisma/schema.prisma first. Generating into
# /install (the site-packages payload copied into every service image)
# means each service's Dockerfile needs no changes or extra PYTHONPATH.
# --generator py_client skips the sibling `prisma-client-js` generator block,
# which needs an npm/Node project we don't have in this Python image.
COPY packages/db/prisma /build/packages/db/prisma
# A normal (non --target) install here, unlike the main dependency install
# above: `pip install --target=` doesn't create the `prisma-client-py`
# console-script entry point, which Prisma's own CLI shells out to as a
# subprocess during generate - it needs to actually be on PATH.
RUN pip install prisma && \
    DATABASE_URL="postgresql://placeholder:placeholder@localhost:5432/placeholder" \
    prisma generate --generator py_client --schema=/build/packages/db/prisma/schema.prisma && \
    rm -rf /install/prisma && \
    cp -r /build/packages/db/generated/python/prisma /install/prisma

# `generate` only produces the client's Python source - the actual query
# engine it calls at runtime is a separate platform-specific Rust binary,
# fetched here into /root/.cache/prisma-python. That cache has to be
# explicitly carried into the runtime stage below (COPY --from=builder),
# since it lives outside /install and nothing else copies it.
RUN prisma py fetch

# Runtime stage
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1: OpenMP runtime used by scikit-learn/xgboost/torch at inference
# time. libxrender1/libxext6: X11 rendering libs that rdkit.Chem.Draw needs
# just to *import* (pulled in transitively via unimol-tools ->
# rdkit.Chem.PandasTools), even though nothing here renders to a display.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libxrender1 libxext6 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all pre-built dependencies
COPY --from=builder /install /usr/local/lib/python3.13/site-packages
# The Prisma Python client's query engine binary (fetched above via
# `prisma py fetch`), without which every DB-touching backend fails at
# startup with BinaryNotFoundError.
COPY --from=builder /root/.cache/prisma-python /root/.cache/prisma-python

# Create app directories
RUN mkdir -p /app/apps /app/packages
