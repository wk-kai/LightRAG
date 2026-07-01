"""MinIO object storage for LightRAG images.

Uploads extracted document images to MinIO and returns public URLs
for storage in images_vdb. Falls back gracefully when MinIO is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from lightrag.utils import logger

_minio_client = None
_minio_bucket = "lightrag-images"
_minio_available: Optional[bool] = None


def _get_minio_client():
    """Lazy-init MinIO client from environment variables."""
    global _minio_client, _minio_available, _minio_bucket

    if _minio_available is not None:
        return _minio_client

    endpoint = os.getenv("MINIO_ENDPOINT", "106.15.102.66:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket = os.getenv("MINIO_BUCKET", "lightrag-images")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    _minio_bucket = bucket

    try:
        from minio import Minio
        from minio.error import S3Error

        _minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        # Ensure bucket exists
        if not _minio_client.bucket_exists(bucket):
            _minio_client.make_bucket(bucket)
            logger.info(f"Created MinIO bucket: {bucket}")

        _minio_available = True
        logger.info(f"MinIO connected: {endpoint}/{bucket}")
        return _minio_client
    except Exception as e:
        _minio_available = False
        logger.warning(f"MinIO unavailable ({e}); falling back to local file paths")
        return None


def upload_image(local_path: str | Path, object_name: Optional[str] = None) -> Optional[str]:
    """Upload an image to MinIO and return its public URL.

    Args:
        local_path: Path to the image file on disk.
        object_name: MinIO object name (auto-generated from path if None).

    Returns:
        Public HTTP URL to the uploaded image, or None if upload fails.
    """
    client = _get_minio_client()
    if client is None:
        return None

    path = Path(local_path)
    if not path.exists():
        logger.warning(f"Image not found for MinIO upload: {local_path}")
        return None

    if object_name is None:
        object_name = path.name

    try:
        content_type = _guess_content_type(path)
        client.fput_object(
            _minio_bucket,
            object_name,
            str(path),
            content_type=content_type,
        )
        # Return logical path, not a full URL — endpoint is resolved at query time
        # from the current MINIO_ENDPOINT env var so the same data works across
        # different hosts (local Docker, remote server, etc.).
        logical_path = f"minio://{_minio_bucket}/{object_name}"
        logger.info(f"Uploaded to MinIO: {logical_path}")
        return logical_path
    except Exception as e:
        logger.warning(f"MinIO upload failed for {local_path}: {e}")
        return None


def delete_image(object_name: str) -> bool:
    """Delete an image from MinIO."""
    client = _get_minio_client()
    if client is None:
        return False
    try:
        client.remove_object(_minio_bucket, object_name)
        return True
    except Exception as e:
        logger.warning(f"MinIO delete failed for {object_name}: {e}")
        return False


def delete_images_by_prefix(prefix: str) -> int:
    """Delete all images with a given object name prefix from MinIO.
    Returns count of deleted objects."""
    client = _get_minio_client()
    if client is None:
        return 0
    count = 0
    try:
        objects = client.list_objects(_minio_bucket, prefix=prefix, recursive=True)
        for obj in objects:
            client.remove_object(_minio_bucket, obj.object_name)
            count += 1
    except Exception as e:
        logger.warning(f"MinIO batch delete failed for prefix {prefix}: {e}")
    return count


def is_minio_available() -> bool:
    """Check if MinIO is connected."""
    _get_minio_client()
    return _minio_available is True


def resolve_minio_url(logical_path: str) -> str:
    """Resolve a ``minio://bucket/object`` logical path to a full HTTP URL.

    Uses the current ``MINIO_ENDPOINT`` env var so the same stored data works
    across different deployments.
    """
    if not logical_path.startswith("minio://"):
        return logical_path
    path = logical_path[len("minio://"):]
    endpoint = os.getenv("MINIO_ENDPOINT", "106.15.102.66:9000")
    return f"http://{endpoint}/{path}"


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")
