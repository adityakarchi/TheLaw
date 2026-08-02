"""Usage Analytics — logs analysis events to S3 and provides dashboard data.

Feature 8: Usage Analytics.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from io import StringIO

import pandas as pd

from src.utils.s3_storage import _get_env, is_s3_configured

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def _get_s3_client():
    """Create S3 client for analytics."""
    return boto3.client(
        "s3",
        aws_access_key_id=_get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_get_env("AWS_SECRET_ACCESS_KEY"),
        region_name=_get_env("AWS_REGION", "ap-south-1"),
    )


def log_analysis_event(event: Dict[str, Any]) -> None:
    """Append a JSON record to the daily analytics log in S3.

    Args:
        event: Dict with fields like timestamp, session_id, pipeline,
               doc_type, file_size_kb, is_legal, risk_level,
               processing_time_ms, chunk_count, language, s3_key.

    Silent failure if S3 is not configured.
    """
    if not is_s3_configured() or not BOTO3_AVAILABLE:
        return

    try:
        bucket = _get_env("S3_BUCKET_NAME")
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"logs/{date_str}.jsonl"

        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.utcnow().isoformat()

        record = json.dumps(event, default=str) + "\n"

        # Try to append to existing file
        s3 = _get_s3_client()
        try:
            existing = s3.get_object(Bucket=bucket, Key=key)
            existing_data = existing["Body"].read().decode("utf-8")
            new_data = existing_data + record
        except ClientError:
            # File doesn't exist yet
            new_data = record

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=new_data.encode("utf-8"),
            ContentType="application/jsonl",
        )

    except Exception as e:
        logger.warning(f"Failed to log analytics event: {e}")


def load_analytics(days: int = 7) -> pd.DataFrame:
    """Read last N days of log files from S3.

    Args:
        days: Number of days of history to load.

    Returns:
        DataFrame with all analytics records.
    """
    if not is_s3_configured() or not BOTO3_AVAILABLE:
        return pd.DataFrame()

    try:
        bucket = _get_env("S3_BUCKET_NAME")
        s3 = _get_s3_client()
        records = []

        for i in range(days):
            date = datetime.utcnow() - timedelta(days=i)
            key = f"logs/{date.strftime('%Y-%m-%d')}.jsonl"

            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                content = obj["Body"].read().decode("utf-8")
                for line in content.strip().split("\n"):
                    if line.strip():
                        records.append(json.loads(line))
            except ClientError:
                continue  # No log file for this day

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", ascending=False)

        return df

    except Exception as e:
        logger.error(f"Failed to load analytics: {e}")
        return pd.DataFrame()


def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from analytics DataFrame.

    Args:
        df: DataFrame from load_analytics().

    Returns:
        Dict with total_analyses, avg_processing_time, legal_percentage,
        risk_distribution, top_languages, daily_counts.
    """
    if df.empty:
        return {
            "total_analyses": 0,
            "avg_processing_time": 0,
            "legal_percentage": 0,
            "risk_distribution": {},
            "top_languages": {},
            "daily_counts": {},
        }

    total = len(df)

    avg_time = 0
    if "processing_time_ms" in df.columns:
        avg_time = int(df["processing_time_ms"].mean())

    legal_pct = 0
    if "is_legal" in df.columns:
        legal_pct = int(df["is_legal"].sum() / total * 100)

    risk_dist = {}
    if "risk_level" in df.columns:
        risk_dist = df["risk_level"].value_counts().to_dict()

    top_langs = {}
    if "language" in df.columns:
        top_langs = df["language"].value_counts().head(5).to_dict()

    daily = {}
    if "timestamp" in df.columns:
        try:
            daily = df.groupby(df["timestamp"].dt.date).size().to_dict()
            daily = {str(k): int(v) for k, v in daily.items()}
        except Exception:
            pass

    return {
        "total_analyses": total,
        "avg_processing_time": avg_time,
        "legal_percentage": legal_pct,
        "risk_distribution": risk_dist,
        "top_languages": top_langs,
        "daily_counts": daily,
    }
