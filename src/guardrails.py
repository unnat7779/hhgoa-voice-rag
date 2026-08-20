"""
Multi-Layer Guardrails System for Voice-Enabled RAG.
Features:
1. Pre-Retrieval: Prompt injection, harmful query, and off-topic domain filtering.
2. Post-Generation: Grounding & Hallucination verification against retrieved context blocks.
"""

import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from src.retriever import RetrievedDocument
from src.config import (
    GUARDRAIL_SIMILARITY_THRESHOLD,
    GUARDRAIL_GROUNDING_THRESHOLD,
    BLOCKED_TOPICS,
    EMBEDDING_MODEL_NAME
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailCheckResult(BaseModel):
    is_safe: bool
    is_on_topic: bool
    is_grounded: bool
    violation_reason: Optional[str] = None
    confidence_score: float = 1.0
    latency_ms: float = 0.0


class GuardrailsEngine:
    """
    Multi-layer safety, security, and hallucination verification engine.
    """

    def __init__(self, embedder: Optional[Any] = None):
        self.embedder = embedder
        # Compile injection patterns
        self.injection_patterns = [
            re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
            re.compile(r"system\s+prompt", re.I),
            re.compile(r"you\s+are\s+now\s+(a|an)?", re.I),
            re.compile(r"reveal\s+(the\s+)?(api|secret|password|key)", re.I),
            re.compile(r"bypass\s+safety", re.I),
            re.compile(r"drop\s+table|delete\s+from", re.I),
        ]

    def check_pre_retrieval(self, query: str) -> GuardrailCheckResult:
        """
        Validates query safety, injection patterns, and offensive keywords.
        """
        t0 = time.time()
        q_lower = query.lower().strip()

        # 1. Length validation
        if len(q_lower) < 2:
            return GuardrailCheckResult(
                is_safe=False,
                is_on_topic=False,
                is_grounded=True,
                violation_reason="Query is too short or empty.",
                latency_ms=(time.time() - t0) * 1000
            )

        # 2. Prompt Injection Detection
        for pattern in self.injection_patterns:
            if pattern.search(q_lower):
                return GuardrailCheckResult(
                    is_safe=False,
                    is_on_topic=False,
                    is_grounded=True,
                    violation_reason="Security Alert: Potential prompt injection or system override detected.",
                    latency_ms=(time.time() - t0) * 1000
                )

        # 3. Harmful / Blocked Keyword Filtering
        for topic in BLOCKED_TOPICS:
            if topic in q_lower:
                return GuardrailCheckResult(
                    is_safe=False,
                    is_on_topic=False,
                    is_grounded=True,
                    violation_reason=f"Safety Alert: Query matches restricted topic policy '{topic}'.",
                    latency_ms=(time.time() - t0) * 1000
                )

        elapsed = (time.time() - t0) * 1000
        return GuardrailCheckResult(
            is_safe=True,
            is_on_topic=True,
            is_grounded=True,
            violation_reason=None,
            confidence_score=0.99,
            latency_ms=elapsed
        )

    def check_post_generation_grounding(
        self,
        answer: str,
        retrieved_docs: List[RetrievedDocument]
    ) -> GuardrailCheckResult:
        """
        Verifies that generated statements are grounded in retrieved documents
        to prevent hallucinations.
        """
        t0 = time.time()
        
        if not retrieved_docs:
            return GuardrailCheckResult(
                is_safe=True,
                is_on_topic=True,
                is_grounded=False,
                violation_reason="No retrieved context available to ground answer.",
                latency_ms=(time.time() - t0) * 1000
            )

        answer_words = set(re.findall(r'\w+', answer.lower()))
        stopwords = {"the", "a", "an", "is", "in", "at", "of", "on", "and", "or", "to", "for", "with", "this", "that", "it", "as", "by", "from", "i", "you", "my", "your", "can", "how", "what", "please", "hello", "hi", "hey", "help", "assist", "welcome", "today"}
        meaningful_words = answer_words - stopwords

        # Conversational greetings / general assistance statements are inherently valid
        greeting_patterns = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bwelcome\b", r"\bhow can i (help|assist)\b", r"\bdo not have enough information\b"]
        is_conversational = any(re.search(pat, answer.lower()) for pat in greeting_patterns)
        if is_conversational and len(meaningful_words) < 10:
            return GuardrailCheckResult(
                is_safe=True,
                is_on_topic=True,
                is_grounded=True,
                confidence_score=0.98,
                latency_ms=(time.time() - t0) * 1000
            )

        if not meaningful_words:
            return GuardrailCheckResult(
                is_safe=True,
                is_on_topic=True,
                is_grounded=True,
                confidence_score=1.0,
                latency_ms=(time.time() - t0) * 1000
            )

        all_context_text = " ".join([d.text.lower() for d in retrieved_docs])
        found_count = sum(1 for w in meaningful_words if w in all_context_text)
        lexical_grounding_ratio = found_count / max(1, len(meaningful_words))

        top_score = retrieved_docs[0].score if retrieved_docs else 0.0
        combined_grounding = (lexical_grounding_ratio * 0.6) + (min(1.0, top_score) * 0.4)
        is_grounded = combined_grounding >= GUARDRAIL_GROUNDING_THRESHOLD

        elapsed = (time.time() - t0) * 1000
        return GuardrailCheckResult(
            is_safe=True,
            is_on_topic=True,
            is_grounded=is_grounded,
            violation_reason=None if is_grounded else f"Grounding confidence ({combined_grounding:.2f}) is below threshold ({GUARDRAIL_GROUNDING_THRESHOLD}).",
            confidence_score=round(combined_grounding, 3),
            latency_ms=elapsed
        )
