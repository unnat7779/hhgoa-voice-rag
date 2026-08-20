"""
Hybrid Multi-Collection Retriever with Reciprocal Rank Fusion (RRF),
Metadata Score Boosting, and Hierarchical Parent-Context Expansion.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import (
    QDRANT_PATH,
    EMBEDDING_MODEL_NAME,
    COLLECTION_SEMANTIC,
    COLLECTION_PASSAGE,
    COLLECTION_SENTENCE,
    COLLECTION_HIERARCHICAL,
    TOP_K_RETRIEVAL,
    RRF_K_CONSTANT,
    METADATA_BOOST_SELECTED,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRATEGY_COLLECTION_MAP = {
    "semantic": [COLLECTION_SEMANTIC],
    "hierarchical": [COLLECTION_HIERARCHICAL],
    "sentence": [COLLECTION_SENTENCE],
    "passage": [COLLECTION_PASSAGE],
    "all": [COLLECTION_SEMANTIC, COLLECTION_PASSAGE, COLLECTION_SENTENCE, COLLECTION_HIERARCHICAL]
}


class RetrievedDocument(BaseModel):
    """
    Standardized schema for retrieved context documents.
    """
    doc_id: str
    text: str
    strategy: str
    score: float
    rrf_score: Optional[float] = 0.0
    source_passage_id: str
    query_id: Optional[Union[str, int]] = None
    query_type: Optional[str] = None
    is_selected: Optional[bool] = False
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    token_count: Optional[int] = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiStrategyRetriever:
    """
    High-performance retriever querying across vector collections,
    merging results via Reciprocal Rank Fusion (RRF) and metadata-aware boosting.
    """

    def __init__(
        self,
        qdrant_path: Optional[str] = QDRANT_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
        embedder: Optional[SentenceTransformer] = None
    ):
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = SentenceTransformer(model_name, device="cpu")
        
        self.client = QdrantClient(path=qdrant_path)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        strategy: str = "all",
        collections: Optional[List[str]] = None,
        apply_rrf: bool = True,
        expand_hierarchical_parents: bool = True,
        apply_metadata_boost: bool = True
    ) -> List[RetrievedDocument]:
        """
        Executes parallel multi-strategy retrieval, applies RRF fusion,
        and expands hierarchical context.
        """
        t0 = time.time()
        
        # 1. Fast query embedding on CPU
        query_vector = self.embedder.encode(
            query, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
        
        if not collections:
            collections = STRATEGY_COLLECTION_MAP.get(strategy, STRATEGY_COLLECTION_MAP["all"])

        # 2. Query target collection(s)
        collection_results: Dict[str, List[Any]] = {}
        for coll in collections:
            try:
                res = self.client.query_points(
                    collection_name=coll,
                    query=query_vector,
                    limit=top_k * 2
                )
                collection_results[coll] = res.points
            except Exception as e:
                logger.warning(f"Failed querying collection {coll}: {e}")
                collection_results[coll] = []

        # 3. Reciprocal Rank Fusion (RRF) & Deduplication
        doc_map: Dict[str, RetrievedDocument] = {}
        rrf_scores: Dict[str, float] = {}

        for coll, points in collection_results.items():
            for rank, point in enumerate(points):
                payload = point.payload or {}
                key = f"{payload.get('source_passage_id')}_{payload.get('chunk_id', point.id)}"
                
                # RRF calculation
                rrf_increment = 1.0 / (RRF_K_CONSTANT + rank + 1)
                if apply_metadata_boost and payload.get("is_selected", False):
                    rrf_increment *= METADATA_BOOST_SELECTED

                rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_increment

                if key not in doc_map:
                    chunk_text = payload.get("text", "")
                    parent_text = payload.get("parent_text")
                    
                    if expand_hierarchical_parents and payload.get("strategy") == "hierarchical_child" and parent_text:
                        effective_text = f"{chunk_text}\n[Context: {parent_text}]"
                    else:
                        effective_text = chunk_text

                    doc_map[key] = RetrievedDocument(
                        doc_id=str(point.id),
                        text=effective_text,
                        strategy=payload.get("strategy", "unknown"),
                        score=float(point.score),
                        source_passage_id=str(payload.get("source_passage_id", "")),
                        query_id=str(payload.get("query_id")) if payload.get("query_id") is not None else None,
                        query_type=payload.get("query_type"),
                        is_selected=bool(payload.get("is_selected", False)),
                        parent_id=str(payload.get("parent_id")) if payload.get("parent_id") is not None else None,
                        parent_text=parent_text,
                        token_count=int(payload.get("token_count", 0)),
                        metadata=payload
                    )

        # 4. Rank by fused score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        top_docs: List[RetrievedDocument] = []
        
        for k in sorted_keys[:top_k]:
            doc = doc_map[k]
            doc.rrf_score = round(rrf_scores[k], 6)
            top_docs.append(doc)

        elapsed = (time.time() - t0) * 1000
        logger.debug(f"MultiStrategyRetriever finished in {elapsed:.2f}ms. Returned {len(top_docs)} docs.")
        return top_docs
