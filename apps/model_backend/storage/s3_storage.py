from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    import boto3
except Exception:  # pragma: no cover - boto3 is optional for local-only runs
    boto3 = None


USE_S3 = os.getenv("USE_S3", "true").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION = os.getenv("S3_REGION")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")


def _client():
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 storage but is not installed")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def build_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


async def upload_file(local_path: Path, key: str, bucket: Optional[str] = None) -> str:
    target_bucket = bucket or S3_BUCKET
    if not USE_S3:
        raise RuntimeError("S3 storage is disabled")
    if not target_bucket:
        raise RuntimeError("S3_BUCKET is required when USE_S3=true")

    client = _client()
    await asyncio.to_thread(client.upload_file, str(local_path), target_bucket, key)
    return build_s3_uri(target_bucket, key)


def upload_file_sync(local_path: Path, key: str, bucket: Optional[str] = None) -> str:
    target_bucket = bucket or S3_BUCKET
    if not USE_S3:
        raise RuntimeError("S3 storage is disabled")
    if not target_bucket:
        raise RuntimeError("S3_BUCKET is required when USE_S3=true")

    client = _client()
    client.upload_file(str(local_path), target_bucket, key)
    return build_s3_uri(target_bucket, key)


async def download_to_temp(uri: str, suffix: str = "") -> Path:
    bucket, key = parse_s3_uri(uri)
    if not USE_S3:
        raise RuntimeError("S3 storage is disabled")

    tmp_dir = Path(tempfile.gettempdir()) / "gene_web_s3"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / Path(key).name
    client = _client()
    await asyncio.to_thread(client.download_file, bucket, key, str(tmp_path))
    if suffix and tmp_path.suffix != suffix:
        target = tmp_path.with_suffix(suffix)
        tmp_path.rename(target)
        return target
    return tmp_path


def download_to_temp_sync(uri: str, suffix: str = "") -> Path:
    bucket, key = parse_s3_uri(uri)
    if not USE_S3:
        raise RuntimeError("S3 storage is disabled")

    tmp_dir = Path(tempfile.gettempdir()) / "gene_web_s3"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / Path(key).name
    client = _client()
    client.download_file(bucket, key, str(tmp_path))
    if suffix and tmp_path.suffix != suffix:
        target = tmp_path.with_suffix(suffix)
        tmp_path.rename(target)
        return target
    return tmp_path
