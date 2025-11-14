import os
import uuid
import csv
import tempfile
import asyncio
from pathlib import Path
from typing import Tuple
from client.db import db

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", "/Users/smarthsood/Desktop/Gene_startup/gene_web/uploads")

if USE_S3:
    import boto3
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    S3_REGION = os.getenv("S3_REGION")
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    )


async def get_csv_shape(path: Path) -> Tuple[int, int]:
    """Count rows and columns for a CSV file at path."""
    def _count():
        rows = 0
        cols = 0
        # open with universal newline support, default encoding
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    cols = len(row)
                rows += 1
        return rows, cols

    return await asyncio.to_thread(_count)


async def save_upload(name,description,file, owner_id: str) -> Tuple[str, str]:
    """
    Save the upload and return (dataset_id, uri).
    - For local storage: saves under DATA_DIR/<owner_id>/<ds_id>/<filename>
    - For S3: uploads to s3://bucket/<owner_id>/<ds_id>/<filename>
      but still uses a local temp file to compute CSV shape safely.
    """
    ds_id = f"ds_{uuid.uuid4().hex[:10]}"
    filename = Path(file.filename).name  # sanitize filename

    # We'll create a local temp file (always) to compute CSV shape reliably.
    # For local storage we will move it into the DATA_DIR; for S3 we upload it.
    tmp_dir = Path(tempfile.gettempdir()) / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{ds_id}_{filename}"

    # Ensure pointer at start and write stream to tmp_path in a thread
    try:
        # file.file may be a SpooledTemporaryFile; ensure pointer at start
        try:
            file.file.seek(0)
        except Exception:
            pass

        # Read file bytes and write to temp path in a thread to avoid blocking
        def _write_tmp():
            # read in chunks to avoid memory spike for huge files
            with open(tmp_path, "wb") as out:
                file.file.seek(0)
                while True:
                    chunk = file.file.read(8192)
                    if not chunk:
                        break
                    out.write(chunk)
            # make sure to reset pointer for future operations (not strictly required)
        await asyncio.to_thread(_write_tmp)

    except Exception as e:
        print(f"Error saving upload to temp file: {e}")
        raise

    # Compute CSV shape (best-effort). If not CSV or error, set to None.
    row_count = None
    col_count = None
    try:
        # naive CSV detection: try to read first few bytes and check for commas/newlines
        # We'll still call get_csv_shape and let it raise if it's not proper CSV.
        row_count, col_count = await get_csv_shape(tmp_path)
    except Exception as e:
        # not a CSV or reading failed; keep None and continue
        print(f"Could not compute CSV shape for {tmp_path}: {e}")
        row_count, col_count = None, None

    # Now decide final storage path / URI
    if USE_S3:
        # upload temp file to S3 (run blocking boto3 in thread)
        key = f"{owner_id}/{ds_id}/{filename}"
        uri = f"s3://{S3_BUCKET}/{key}"
        try:
            await asyncio.to_thread(s3.upload_file, str(tmp_path), S3_BUCKET, key)
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            # optionally cleanup tmp_path here
            raise
        # optional: remove tmp file after successful upload
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        final_path = uri  # store S3 uri in DB

    else:
        # move temp file to permanent DATA_DIR path
        base = Path(DATA_DIR) / str(owner_id) / ds_id
        base.mkdir(parents=True, exist_ok=True)
        dest = base / filename
        try:
            await asyncio.to_thread(tmp_path.replace, dest)
        except Exception as e:
            # fallback to copy if replace fails
            await asyncio.to_thread(tmp_path.rename, dest)
        final_path = str(dest)

    # Upsert into DB (your Dataset model uses id: String @id and a required relation to User)
    try:
        await db.dataset.upsert(
            where={"id": ds_id},
            data={
                "create": {
                    "id": ds_id,
                    "filePath": final_path,
                    "user": {"connect": {"id": int(owner_id)}},
                    "name": name,
                    "description": description,
                    "rowCount": row_count,
                    "columnCount": col_count,
                },
                "update": {
                    "filePath": final_path,
                    "rowCount": row_count,
                    "columnCount": col_count,
                },
            },
        )
    except Exception as e:
        # Keep helpful debug info
        print(f"Error upserting dataset in DB: {e}")
        # If you want to fail the request, raise here; else return what we have.
        raise

    return ds_id, final_path
