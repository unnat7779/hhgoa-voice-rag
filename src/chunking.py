"""
Multi-strategy chunking engine for RAG.
Implements:
1. Passage-Level (Baseline)
2. Semantic Chunking (Cosine similarity breakpoint detection with high-throughput batching)
3. Sentence-Level with Context Window Overlap
4. Recursive Character / Word Boundary Chunking with Overlap
5. Hierarchical Parent-Child Chunking
6. Metadata-Aware Enrichment on all chunks
"""

import re
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import (
    SEMANTIC_SIMILARITY_THRESHOLD,
    SENTENCE_WINDOW_SIZE,
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    source_passage_id: str
    query_id: Optional[int] = None
    query_type: str = "unknown"
    is_selected: bool = False
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    position_index: int = 0
    token_count: int = 0
    char_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_into_sentences(text: str) -> List[str]:
    """
    Robust regex-based sentence boundary splitter.
    """
    if not text:
        return []
    text = " ".join(text.split())
    pattern = r'(?<=[.?!])\s+(?=[A-Z0-9"\'\(\[])'
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def count_tokens(text: str) -> int:
    return len(text.split())


class PassageChunker:
    """Strategy 1: Passage-Level baseline."""

    @staticmethod
    def chunk(passage: Dict[str, Any]) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []
        
        cid = f"chk_pass_{passage.get('passage_id', uuid.uuid4().hex[:8])}"
        return [
            Chunk(
                chunk_id=cid,
                text=text,
                strategy="passage_level",
                source_passage_id=passage.get("passage_id", ""),
                query_id=passage.get("query_id"),
                query_type=passage.get("query_type", "unknown"),
                is_selected=passage.get("is_selected", False),
                parent_id=passage.get("passage_id", ""),
                parent_text=text,
                position_index=0,
                token_count=count_tokens(text),
                char_length=len(text),
                metadata={
                    "translated_text": passage.get("translated_text", ""),
                    "source_lang": passage.get("source_lang", "eng_Latn"),
                    "target_lang": passage.get("target_lang", "hin_Deva"),
                }
            )
        ]


class SentenceWindowChunker:
    """Strategy 2: Sentence-Level with Context Window Overlap (±k sentences)."""

    def __init__(self, window_size: int = SENTENCE_WINDOW_SIZE):
        self.window_size = window_size

    def chunk(self, passage: Dict[str, Any]) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []
        
        sentences = split_into_sentences(text)
        if not sentences:
            return []
        
        chunks: List[Chunk] = []
        p_id = passage.get("passage_id", uuid.uuid4().hex[:8])

        for i, target_sentence in enumerate(sentences):
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(sentences), i + self.window_size + 1)
            window_text = " ".join(sentences[start_idx:end_idx])

            cid = f"chk_sent_{p_id}_{i}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=window_text,
                    strategy="sentence_window",
                    source_passage_id=p_id,
                    query_id=passage.get("query_id"),
                    query_type=passage.get("query_type", "unknown"),
                    is_selected=passage.get("is_selected", False),
                    parent_id=p_id,
                    parent_text=text,
                    position_index=i,
                    token_count=count_tokens(window_text),
                    char_length=len(window_text),
                    metadata={
                        "target_sentence": target_sentence,
                        "window_start": start_idx,
                        "window_end": end_idx,
                        "total_sentences": len(sentences)
                    }
                )
            )
        return chunks


class RecursiveOverlapChunker:
    """Strategy 3: Recursive Character / Word Boundary with Configurable Overlap."""

    def __init__(self, max_tokens: int = RECURSIVE_CHUNK_SIZE, overlap_tokens: int = RECURSIVE_CHUNK_OVERLAP):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, passage: Dict[str, Any]) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []
        
        p_id = passage.get("passage_id", uuid.uuid4().hex[:8])
        words = text.split()
        if len(words) <= self.max_tokens:
            cid = f"chk_rec_{p_id}_0"
            return [
                Chunk(
                    chunk_id=cid,
                    text=text,
                    strategy="recursive_overlap",
                    source_passage_id=p_id,
                    query_id=passage.get("query_id"),
                    query_type=passage.get("query_type", "unknown"),
                    is_selected=passage.get("is_selected", False),
                    parent_id=p_id,
                    parent_text=text,
                    position_index=0,
                    token_count=len(words),
                    char_length=len(text),
                    metadata={"step": 0, "overlap": 0}
                )
            ]

        chunks: List[Chunk] = []
        step = max(1, self.max_tokens - self.overlap_tokens)
        idx = 0
        for pos in range(0, len(words), step):
            chunk_words = words[pos:pos + self.max_tokens]
            if not chunk_words:
                continue
            chunk_str = " ".join(chunk_words)
            cid = f"chk_rec_{p_id}_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=chunk_str,
                    strategy="recursive_overlap",
                    source_passage_id=p_id,
                    query_id=passage.get("query_id"),
                    query_type=passage.get("query_type", "unknown"),
                    is_selected=passage.get("is_selected", False),
                    parent_id=p_id,
                    parent_text=text,
                    position_index=idx,
                    token_count=len(chunk_words),
                    char_length=len(chunk_str),
                    metadata={"word_start": pos, "word_end": pos + len(chunk_words)}
                )
            )
            idx += 1
            if pos + self.max_tokens >= len(words):
                break
        return chunks


class HierarchicalParentChildChunker:
    """
    Strategy 4: Hierarchical Parent-Child Chunking.
    """

    @staticmethod
    def chunk(passage: Dict[str, Any]) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []
        
        sentences = split_into_sentences(text)
        p_id = passage.get("passage_id", uuid.uuid4().hex[:8])
        chunks: List[Chunk] = []

        for idx, sentence in enumerate(sentences):
            cid = f"chk_hier_child_{p_id}_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=sentence,
                    strategy="hierarchical_child",
                    source_passage_id=p_id,
                    query_id=passage.get("query_id"),
                    query_type=passage.get("query_type", "unknown"),
                    is_selected=passage.get("is_selected", False),
                    parent_id=p_id,
                    parent_text=text,
                    position_index=idx,
                    token_count=count_tokens(sentence),
                    char_length=len(sentence),
                    metadata={
                        "is_child": True,
                        "parent_token_count": count_tokens(text),
                        "parent_char_length": len(text)
                    }
                )
            )
        return chunks


class SemanticChunker:
    """
    Strategy 5: Semantic Chunking with fast vectorized boundary grouping.
    """

    def __init__(self, similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def chunk_passage_with_embeddings(
        self, 
        passage: Dict[str, Any], 
        sentence_embeddings: List[np.ndarray], 
        sentences: List[str]
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []
        
        p_id = passage.get("passage_id", uuid.uuid4().hex[:8])
        if len(sentences) <= 2:
            cid = f"chk_sem_{p_id}_0"
            return [
                Chunk(
                    chunk_id=cid,
                    text=text,
                    strategy="semantic",
                    source_passage_id=p_id,
                    query_id=passage.get("query_id"),
                    query_type=passage.get("query_type", "unknown"),
                    is_selected=passage.get("is_selected", False),
                    parent_id=p_id,
                    parent_text=text,
                    position_index=0,
                    token_count=count_tokens(text),
                    char_length=len(text),
                    metadata={"breakpoint_count": 0}
                )
            ]

        # Calculate consecutive cosine similarities
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = float(np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]))
            similarities.append(sim)

        mean_sim = float(np.mean(similarities)) if similarities else 1.0
        effective_threshold = min(self.similarity_threshold, mean_sim * 0.9)

        grouped_chunks = []
        current_group = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < effective_threshold and count_tokens(" ".join(current_group)) >= 25:
                grouped_chunks.append(" ".join(current_group))
                current_group = [sentences[i + 1]]
            else:
                current_group.append(sentences[i + 1])
        if current_group:
            grouped_chunks.append(" ".join(current_group))

        result_chunks: List[Chunk] = []
        for idx, g_text in enumerate(grouped_chunks):
            cid = f"chk_sem_{p_id}_{idx}"
            result_chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=g_text,
                    strategy="semantic",
                    source_passage_id=p_id,
                    query_id=passage.get("query_id"),
                    query_type=passage.get("query_type", "unknown"),
                    is_selected=passage.get("is_selected", False),
                    parent_id=p_id,
                    parent_text=text,
                    position_index=idx,
                    token_count=count_tokens(g_text),
                    char_length=len(g_text),
                    metadata={
                        "sentences_in_chunk": len(g_text.split(".")),
                        "mean_similarity": mean_sim
                    }
                )
            )
        return result_chunks


class MultiStrategyChunkingEngine:
    """
    Orchestrates all 5 chunking strategies with high-speed vectorized sentence batching.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.passage_chunker = PassageChunker()
        self.sentence_chunker = SentenceWindowChunker()
        self.recursive_chunker = RecursiveOverlapChunker()
        self.semantic_chunker = SemanticChunker()
        self.hierarchical_chunker = HierarchicalParentChildChunker()

    def process_all_passages(
        self, 
        passages: List[Dict[str, Any]], 
        embedder=None
    ) -> Dict[str, List[Chunk]]:
        active_embedder = embedder or self.embedder
        
        categorized: Dict[str, List[Chunk]] = {
            "passage_level": [],
            "sentence_window": [],
            "recursive_overlap": [],
            "semantic": [],
            "hierarchical_child": [],
        }

        logger.info(f"Extracting sentences and applying non-embedding strategies for {len(passages)} passages...")
        all_sentences: List[str] = []
        passage_sentence_slices: List[Tuple[int, int, List[str]]] = []

        for p in passages:
            # Standard chunkers
            categorized["passage_level"].extend(self.passage_chunker.chunk(p))
            categorized["sentence_window"].extend(self.sentence_chunker.chunk(p))
            categorized["recursive_overlap"].extend(self.recursive_chunker.chunk(p))
            categorized["hierarchical_child"].extend(self.hierarchical_chunker.chunk(p))

            # Prepare for semantic chunking
            p_text = p.get("text", "").strip()
            sents = split_into_sentences(p_text)
            start_idx = len(all_sentences)
            all_sentences.extend(sents)
            end_idx = len(all_sentences)
            passage_sentence_slices.append((start_idx, end_idx, sents))

        if active_embedder is not None and all_sentences:
            logger.info(f"Batch embedding {len(all_sentences)} sentences for semantic boundary detection...")
            all_embeddings = active_embedder.encode(
                all_sentences, 
                batch_size=512, 
                show_progress_bar=True, 
                normalize_embeddings=True
            )

            for idx, p in enumerate(passages):
                start_idx, end_idx, sents = passage_sentence_slices[idx]
                p_embs = all_embeddings[start_idx:end_idx]
                sem_chunks = self.semantic_chunker.chunk_passage_with_embeddings(p, p_embs, sents)
                categorized["semantic"].extend(sem_chunks)
        else:
            categorized["semantic"] = categorized["passage_level"]

        logger.info(
            f"Chunking complete. Stats: passage_level={len(categorized['passage_level'])}, "
            f"sentence_window={len(categorized['sentence_window'])}, "
            f"recursive_overlap={len(categorized['recursive_overlap'])}, "
            f"semantic={len(categorized['semantic'])}, "
            f"hierarchical_child={len(categorized['hierarchical_child'])}"
        )
        return categorized
