"""Sentence Transformers embedding wrapper for semantic similarity.

Based on SentenceTransformers semantic similarity examples:
https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html

Using cosine similarity because SBERT embeddings are L2-normalized
and this is the standard approach recommended by SBERT.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Lazy-loaded SentenceTransformer wrapper."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Encode text(s) into embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([])
        return self.model.encode(list(texts), convert_to_numpy=True)

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compare two texts using cosine similarity on their embeddings."""
        if not text_a.strip() or not text_b.strip():
            return 0.0

        embeddings = self.encode([text_a, text_b])
        return self.cosine_similarity_from_vectors(embeddings[0], embeddings[1])

    def cosine_similarity_from_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two precomputed embedding vectors."""
        similarity = cos_sim(vec_a, vec_b).item()
        return float(max(0.0, min(1.0, similarity)))

    def batch_cosine_similarities(self, anchors: list[str], targets: list[str]) -> list[float]:
        """
        Encode all texts once, then compute cosine similarity for each anchor-target pair.

        anchors[i] is compared against targets[i].
        """
        if not anchors or not targets or len(anchors) != len(targets):
            return [0.0] * len(anchors)

        all_texts = anchors + targets
        embeddings = self.encode(all_texts)
        n = len(anchors)
        scores: list[float] = []

        for i in range(n):
            if not anchors[i].strip() or not targets[i].strip():
                scores.append(0.0)
            else:
                scores.append(self.cosine_similarity_from_vectors(embeddings[i], embeddings[n + i]))

        return scores


_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return a shared EmbeddingModel instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def warmup_embedding_model() -> None:
    """Load the embedding model ahead of first user request."""
    get_embedding_model().encode("warmup")
