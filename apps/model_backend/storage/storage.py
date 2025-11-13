import os, uuid
from pathlib import Path
from typing import Tuple
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
USE_S3 = os.getenv("USE_S3").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", "/Users/smarthsood/Desktop/Gene_startup/gene_web/uploads")

print(f"Storage Config - USE_S3: {USE_S3}, DATA_DIR: {DATA_DIR}")
if USE_S3:
    import boto3
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    S3_REGION = os.getenv("S3_REGION")
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT_URL, region_name=S3_REGION,
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"))


def save_upload(file, owner_id: str) -> Tuple[str, str]:
    """Return (dataset_id, uri)."""
    ds_id = f"ds_{uuid.uuid4().hex[:10]}"
    filename = file.filename
    if USE_S3:
        key = f"{owner_id}/{ds_id}/{filename}"
        s3.upload_fileobj(file.file, S3_BUCKET, key)
        return ds_id, f"s3://{S3_BUCKET}/{key}"
    else:
        base = Path(DATA_DIR) / owner_id / ds_id
        base.mkdir(parents=True, exist_ok=True)
        path = base / filename
        with open(path, "wb") as out:
            out.write(file.file.read())
        return ds_id, str(path)