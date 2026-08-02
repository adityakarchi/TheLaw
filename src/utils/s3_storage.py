"""S3 Storage Utility — handles uploading and downloading legal documents from AWS S3.

Usage:
    from src.utils.s3_storage import upload_pdf_to_s3, list_uploaded_documents
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import so the app still works without boto3 installed
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. S3 features disabled. Run: pip install boto3")


def _get_s3_client():
    """Create and return a boto3 S3 client using env credentials."""
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 is not installed. Run: pip install boto3")

    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def is_s3_configured() -> bool:
    """Return True if all required AWS env vars are set."""
    return all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("S3_BUCKET_NAME"),
        BOTO3_AVAILABLE,
    ])


def upload_pdf_to_s3(
    file_bytes: bytes,
    filename: str,
    prefix: str = "uploads",
) -> Optional[str]:
    """Upload a PDF file to S3 and return the S3 object key.

    Args:
        file_bytes: Raw bytes of the file.
        filename:   Original filename (e.g. "contract.pdf").
        prefix:     S3 folder prefix (default: "uploads").

    Returns:
        S3 object key string if successful, None otherwise.
    """
    if not is_s3_configured():
        logger.warning("S3 not configured — skipping upload.")
        return None

    bucket = os.getenv("S3_BUCKET_NAME")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Sanitize filename: replace spaces
    safe_name = filename.replace(" ", "_")
    key = f"{prefix}/{timestamp}_{safe_name}"

    try:
        s3 = _get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_bytes,
            ContentType="application/pdf",
            ServerSideEncryption="AES256",          # encrypt at rest
            Metadata={
                "original-filename": filename,
                "uploaded-at": datetime.utcnow().isoformat(),
            },
        )
        logger.info(f"Uploaded to s3://{bucket}/{key}")
        return key

    except NoCredentialsError:
        logger.error("AWS credentials not found or invalid.")
        return None
    except ClientError as e:
        logger.error(f"S3 upload failed: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        logger.error(f"Unexpected S3 error: {e}")
        return None


def upload_text_to_s3(
    text: str,
    filename: str,
    prefix: str = "uploads",
) -> Optional[str]:
    """Upload plain text content to S3 and return the S3 object key.

    Args:
        text:     The text content to store.
        filename: Desired filename (e.g. "contract.txt").
        prefix:   S3 folder prefix (default: "uploads").

    Returns:
        S3 object key string if successful, None otherwise.
    """
    if not is_s3_configured():
        logger.warning("S3 not configured — skipping upload.")
        return None

    bucket = os.getenv("S3_BUCKET_NAME")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = filename.replace(" ", "_")
    key = f"{prefix}/{timestamp}_{safe_name}"

    try:
        s3 = _get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
            ServerSideEncryption="AES256",
            Metadata={
                "original-filename": filename,
                "uploaded-at": datetime.utcnow().isoformat(),
            },
        )
        logger.info(f"Uploaded text to s3://{bucket}/{key}")
        return key

    except NoCredentialsError:
        logger.error("AWS credentials not found or invalid.")
        return None
    except ClientError as e:
        logger.error(f"S3 upload failed: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        logger.error(f"Unexpected S3 error: {e}")
        return None


def list_uploaded_documents(prefix: str = "uploads", max_items: int = 50) -> list[dict]:
    """List recently uploaded documents from S3.

    Returns:
        List of dicts with keys: key, filename, size_kb, uploaded_at
    """
    if not is_s3_configured():
        return []

    bucket = os.getenv("S3_BUCKET_NAME")

    try:
        s3 = _get_s3_client()
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_items,
        )

        items = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            # Strip timestamp prefix from display name (format: YYYYMMDD_HHMMSS_name)
            parts = filename.split("_", 2)
            display_name = parts[2] if len(parts) == 3 else filename

            items.append({
                "key": key,
                "filename": display_name,
                "size_kb": round(obj["Size"] / 1024, 1),
                "uploaded_at": obj["LastModified"].strftime("%Y-%m-%d %H:%M UTC"),
            })

        # Most recent first
        items.sort(key=lambda x: x["uploaded_at"], reverse=True)
        return items

    except Exception as e:
        logger.error(f"Failed to list S3 objects: {e}")
        return []


def get_s3_console_url(key: str) -> str:
    """Return the AWS console URL for a given S3 object key."""
    bucket = os.getenv("S3_BUCKET_NAME", "")
    region = os.getenv("AWS_REGION", "us-east-1")
    return f"https://s3.console.aws.amazon.com/s3/object/{bucket}?region={region}&prefix={key}"
