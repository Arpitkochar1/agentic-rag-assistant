"""Lightweight ONNX embeddings (no PyTorch) -> small image, fits Streamlit Cloud."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(list(self._model.passage_embed(list(texts))), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype="float32")
