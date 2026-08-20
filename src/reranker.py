"""
Cross-Encoder and Fast Cosine Re-ranker module.
Re-orders retrieved candidates to maximize relevance precision for answer generation.
"""

import time
import logging
from typing import List, Optional
from src.retriever import RetrievedDocument
from src.config import TOP_K_RERANK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReRanker:
    """
    Reranker supporting both high-speed score calibration and Cross-Encoder re-ranking.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._cross_encoder = None

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}...")
                self._cross_encoder = CrossEncoder(self.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder: {e}. Falling back to score fusion.")
                self._cross_encoder = False

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = TOP_K_RERANK,
        use_cross_encoder: bool = False
    ) -> List[RetrievedDocument]:
        """
        Re-ranks top candidate documents.
        If use_cross_encoder is True and model is available, computes joint query-passage attention.
        Otherwise applies calibrated RRF + Cosine score weighting.
        """
        if not documents:
            return []

        t0 = time.time()

        if use_cross_encoder and self.model_name:
            self._load_cross_encoder()
            if self._cross_encoder:
                pairs = [[query, doc.text] for doc in documents]
                scores = self._cross_encoder.predict(pairs)
                for i, doc in enumerate(documents):
                    doc.score = float(scores[i])
                sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
                elapsed = (time.time() - t0) * 1000
                logger.debug(f"CrossEncoder reranked {len(documents)} docs in {elapsed:.2f}ms.")
                return sorted_docs[:top_k]

        # Fast heuristic reranking (Vector Cosine + RRF score blend + Metadata weighting)
        for doc in documents:
            base_score = (doc.score * 0.4) + ((doc.rrf_score or 0.0) * 100.0 * 0.6)
            if doc.is_selected:
                base_score += 0.15
            doc.score = round(base_score, 4)

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        return sorted_docs[:top_k]
