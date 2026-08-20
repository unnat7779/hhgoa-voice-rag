"""
Vector Database Indexer using Qdrant and Sentence-Transformers.
Manages collection creation, batch embedding, and payload indexing.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from src.config import (
    QDRANT_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    COLLECTION_SEMANTIC,
    COLLECTION_PASSAGE,
    COLLECTION_SENTENCE,
    COLLECTION_HIERARCHICAL,
    MAX_PASSAGES_INDEX,
)
from src.chunking import Chunk, MultiStrategyChunkingEngine
from src.data_loader import get_or_create_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize CPU threading for Apple Silicon M-series
torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


class QdrantIndexer:
    """
    Qdrant vector database indexer managing embedding creation and collection upserts.
    """

    def __init__(self, path: Optional[str] = QDRANT_PATH, model_name: str = EMBEDDING_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading embedding model: {model_name} on {device}...")
        self.embedder = SentenceTransformer(model_name, device=device)

        logger.info(f"Initializing Qdrant client at: {path}")
        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

    def create_collections(self, force_recreate: bool = False):
        """
        Creates all required collections in Qdrant with Cosine distance.
        """
        collections = [
            COLLECTION_SEMANTIC,
            COLLECTION_PASSAGE,
            COLLECTION_SENTENCE,
            COLLECTION_HIERARCHICAL
        ]

        existing = [c.name for c in self.client.get_collections().collections]
        logger.info(f"Existing collections: {existing}")

        for coll_name in collections:
            if coll_name in existing:
                if force_recreate:
                    logger.info(f"Recreating collection {coll_name}...")
                    self.client.delete_collection(collection_name=coll_name)
                else:
                    logger.info(f"Collection {coll_name} already exists. Skipping creation.")
                    continue

            logger.info(f"Creating collection {coll_name} (dim={EMBEDDING_DIM})...")
            self.client.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=1000
                )
            )

    def index_chunks(
        self, 
        collection_name: str, 
        chunks: List[Chunk], 
        batch_size: int = 256
    ) -> int:
        """
        Embeds chunks in batches and upserts points with full metadata payload into Qdrant.
        """
        if not chunks:
            logger.warning(f"No chunks provided for collection {collection_name}")
            return 0

        logger.info(f"Indexing {len(chunks)} chunks into '{collection_name}' (batch_size={batch_size})...")
        start_time = time.time()
        total_points = 0

        texts = [c.text for c in chunks]
        
        # High-speed batch encoding on CPU vector SIMD
        embeddings = self.embedder.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            points = []
            for j, chunk in enumerate(batch_chunks):
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}_{chunk.chunk_id}").hex
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "strategy": chunk.strategy,
                    "source_passage_id": chunk.source_passage_id,
                    "query_id": chunk.query_id,
                    "query_type": chunk.query_type,
                    "is_selected": chunk.is_selected,
                    "parent_id": chunk.parent_id,
                    "parent_text": chunk.parent_text,
                    "position_index": chunk.position_index,
                    "token_count": chunk.token_count,
                    "char_length": chunk.char_length,
                    **chunk.metadata
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=batch_embeddings[j].tolist(),
                        payload=payload
                    )
                )

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            total_points += len(points)

        elapsed = time.time() - start_time
        logger.info(f"Indexed {total_points} chunks into '{collection_name}' in {elapsed:.2f}s ({total_points / max(0.01, elapsed):.1f} chunks/sec).")
        return total_points

    def index_all_strategies(
        self, 
        passages: List[Dict[str, Any]], 
        force_recreate: bool = False
    ) -> Dict[str, int]:
        """
        Runs multi-strategy chunking and indexes each category into its respective Qdrant collection.
        """
        self.create_collections(force_recreate=force_recreate)

        chunking_engine = MultiStrategyChunkingEngine(embedder=self.embedder)
        logger.info("Executing multi-strategy chunking across passages...")
        categorized_chunks = chunking_engine.process_all_passages(passages, embedder=self.embedder)

        stats = {}

        # 1. Index passage-level
        stats[COLLECTION_PASSAGE] = self.index_chunks(
            COLLECTION_PASSAGE, 
            categorized_chunks["passage_level"]
        )

        # 2. Index semantic chunks
        stats[COLLECTION_SEMANTIC] = self.index_chunks(
            COLLECTION_SEMANTIC, 
            categorized_chunks["semantic"]
        )

        # 3. Index sentence window & recursive chunks
        sentence_and_recursive = categorized_chunks["sentence_window"] + categorized_chunks["recursive_overlap"]
        stats[COLLECTION_SENTENCE] = self.index_chunks(
            COLLECTION_SENTENCE, 
            sentence_and_recursive
        )

        # 4. Index hierarchical child chunks
        stats[COLLECTION_HIERARCHICAL] = self.index_chunks(
            COLLECTION_HIERARCHICAL, 
            categorized_chunks["hierarchical_child"]
        )

        logger.info(f"Completed indexing across all collections: {stats}")
        return stats


def run_pipeline_indexing(max_passages: int = MAX_PASSAGES_INDEX, force_recreate: bool = True):
    """
    End-to-end execution function for Phase 1.
    """
    logger.info(f"Starting Phase 1 Indexing Pipeline (max_passages={max_passages})...")
    passages, queries = get_or_create_corpus(max_passages=max_passages)
    logger.info(f"Corpus loaded: {len(passages)} passages, {len(queries)} queries.")

    indexer = QdrantIndexer(device="cpu")
    stats = indexer.index_all_strategies(passages, force_recreate=force_recreate)
    logger.info(f"Phase 1 Indexing Pipeline Finished Successfully! Stats: {stats}")
    return stats


if __name__ == "__main__":
    run_pipeline_indexing(max_passages=2000, force_recreate=True)
