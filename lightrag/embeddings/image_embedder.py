"""CLIP-based image embedding for LightRAG.

Lazy-loads the CLIP model on first use and provides a simple interface
for generating visual embeddings of extracted document images.
"""

from __future__ import annotations

import asyncio
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from lightrag.utils import logger

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None
_model_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def _load_clip_model():
    """Load CLIP ViT-B/32 model (lazy, cached)."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    import pipmaster as pm

    if not pm.is_installed("transformers"):
        pm.install("transformers")
    if not pm.is_installed("torch"):
        pm.install("torch")
    if not pm.is_installed("pillow"):
        pm.install("pillow")

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_name = os.getenv("IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
    logger.info(f"Loading CLIP model: {model_name}")
    _clip_model = CLIPModel.from_pretrained(model_name)
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    if torch.cuda.is_available():
        _clip_model = _clip_model.to("cuda")
    _clip_model.eval()
    return _clip_model, _clip_processor


def _image_to_embedding(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to a 512-dim CLIP embedding (sync)."""
    from PIL import Image

    model, processor = _load_clip_model()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    import torch

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    # Handle both legacy (tensor) and new (BaseModelOutputWithPooling) return types
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    elif hasattr(features, "image_embeds"):
        features = features.image_embeds
    vec = features.cpu().numpy().flatten()
    vec = vec / np.linalg.norm(vec)
    return vec


async def embed_images(image_paths: List[str]) -> List[np.ndarray]:
    """Generate CLIP embeddings for a list of image file paths.

    Args:
        image_paths: List of absolute paths to image files.

    Returns:
        List of 512-dim normalized numpy arrays, one per image.
    """
    async with _model_lock:
        _load_clip_model()

    embeddings = []

    def _load_and_embed(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Image not found: {path}")
            return np.zeros(512, dtype=np.float32)
        try:
            raw = p.read_bytes()
            if not raw:
                return np.zeros(512, dtype=np.float32)
            return _image_to_embedding(raw)
        except Exception as e:
            logger.warning(f"Failed to embed image {path}: {e}")
            return np.zeros(512, dtype=np.float32)

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(_executor, _load_and_embed, p) for p in image_paths]
    embeddings = await asyncio.gather(*tasks)
    return list(embeddings)


def clip_embedding_dim() -> int:
    """Return the CLIP embedding dimension (512 for ViT-B/32)."""
    return 512
