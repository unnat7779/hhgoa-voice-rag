"""
Structured Model Harness for Voice-Enabled RAG Pipeline.
Orchestrates End-to-End workflow:
Voice/Text Query -> Pre-Guardrail -> Multi-Strategy Retrieval -> Re-ranking -> LLM Generation -> Post-Grounding -> Structured Response.
Includes granular millisecond latency breakdown and tool orchestration.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from src.retriever import MultiStrategyRetriever, RetrievedDocument
from src.reranker import ReRanker
from src.llm_client import GroqLLMClient, LLMResponse
from src.guardrails import GuardrailsEngine, GuardrailCheckResult
from src.config import TOP_K_RETRIEVAL, TOP_K_RERANK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyProfile(BaseModel):
    stt_ms: float = 0.0
    pre_guardrail_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_generation_ms: float = 0.0
    post_guardrail_ms: float = 0.0
    total_pipeline_ms: float = 0.0


class CitationInfo(BaseModel):
    doc_index: int
    source_passage_id: str
    strategy: str
    score: float
    is_selected: bool
    snippet: str


class RAGRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    audio_base64: Optional[str] = None
    strategy: Optional[str] = "all"
    use_cross_encoder: bool = False
    top_k: int = TOP_K_RETRIEVAL
    model: Optional[str] = None


class RAGResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    is_safe: bool = True
    is_grounded: bool = True
    violation_reason: Optional[str] = None
    citations: List[CitationInfo] = Field(default_factory=list)
    retrieved_count: int = 0
    latency: LatencyProfile
    model_used: str = "llama-3.1-8b-instant"
    used_fallback: bool = False


class StructuredModelHarness:
    """
    Production-grade model harness orchestrating guardrails, retrieval,
    tool execution, and structured generation.
    """

    def __init__(
        self,
        retriever: Optional[MultiStrategyRetriever] = None,
        reranker: Optional[ReRanker] = None,
        llm_client: Optional[GroqLLMClient] = None,
        guardrails: Optional[GuardrailsEngine] = None
    ):
        logger.info("Initializing Structured Model Harness...")
        self.retriever = retriever or MultiStrategyRetriever()
        self.reranker = reranker or ReRanker()
        self.llm_client = llm_client or GroqLLMClient()
        self.guardrails = guardrails or GuardrailsEngine(embedder=self.retriever.embedder)
        
        # Register available tools
        self.tools: Dict[str, Callable] = {
            "lookup_passage": self._tool_lookup_passage,
            "get_corpus_stats": self._tool_get_corpus_stats,
        }

    def _tool_lookup_passage(self, passage_id: str) -> Dict[str, Any]:
        """Tool to inspect full raw passage payload by ID."""
        return {"passage_id": passage_id, "status": "lookup_ready"}

    def _tool_get_corpus_stats(self) -> Dict[str, Any]:
        """Tool to return collection sizes and system status."""
        cols = self.retriever.client.get_collections().collections
        return {"active_collections": [c.name for c in cols], "status": "healthy"}

    def run_pipeline(
        self,
        request: RAGRequest,
        stt_latency_ms: float = 0.0
    ) -> RAGResponse:
        """
        Executes the full RAG pipeline with high-precision latency profiling and fault tolerance.
        """
        request_id = str(uuid.uuid4())[:8]
        t_pipeline_start = time.time()
        latency = LatencyProfile(stt_ms=stt_latency_ms)

        query = request.query.strip()
        logger.info(f"[{request_id}] Processing Query: '{query}' (Strategy: {request.strategy})")

        # -------------------------------------------------------------
        # Step 1: Pre-Retrieval Guardrail (Prompt Injection / Safety)
        # -------------------------------------------------------------
        pre_guard = self.guardrails.check_pre_retrieval(query)
        latency.pre_guardrail_ms = pre_guard.latency_ms

        if not pre_guard.is_safe:
            total_ms = (time.time() - t_pipeline_start) * 1000 + stt_latency_ms
            latency.total_pipeline_ms = round(total_ms, 2)
            logger.warning(f"[{request_id}] Pre-retrieval guardrail triggered: {pre_guard.violation_reason}")
            return RAGResponse(
                request_id=request_id,
                query=query,
                answer=f"I cannot process this request. {pre_guard.violation_reason}",
                is_safe=False,
                is_grounded=True,
                violation_reason=pre_guard.violation_reason,
                citations=[],
                retrieved_count=0,
                latency=latency,
                model_used="guardrail-filter",
                used_fallback=False
            )

        # -------------------------------------------------------------
        # Step 2: Multi-Strategy Vector Retrieval
        # -------------------------------------------------------------
        t0 = time.time()
        retrieved_docs: List[RetrievedDocument] = self.retriever.retrieve(
            query=query,
            top_k=request.top_k,
            strategy=request.strategy or "all",
            apply_rrf=True,
            expand_hierarchical_parents=True,
            apply_metadata_boost=True
        )
        latency.retrieval_ms = round((time.time() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Step 3: Candidate Re-ranking
        # -------------------------------------------------------------
        t0 = time.time()
        reranked_docs: List[RetrievedDocument] = self.reranker.rerank(
            query=query,
            documents=retrieved_docs,
            top_k=TOP_K_RERANK,
            use_cross_encoder=request.use_cross_encoder
        )
        latency.rerank_ms = round((time.time() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Step 4: Answer Generation with LLM
        # -------------------------------------------------------------
        t0 = time.time()
        llm_res: LLMResponse = self.llm_client.generate_answer(
            query=query,
            documents=reranked_docs,
            model=request.model
        )
        latency.llm_generation_ms = round((time.time() - t0) * 1000, 2)

        # -------------------------------------------------------------
        # Step 5: Post-Generation Grounding & Hallucination Guardrail
        # -------------------------------------------------------------
        post_guard = self.guardrails.check_post_generation_grounding(
            answer=llm_res.answer,
            retrieved_docs=reranked_docs
        )
        latency.post_guardrail_ms = round(post_guard.latency_ms, 2)

        # Build citation cards
        citation_cards = []
        for i, doc in enumerate(reranked_docs, 1):
            snippet_text = doc.text[:150] + "..." if len(doc.text) > 150 else doc.text
            citation_cards.append(
                CitationInfo(
                    doc_index=i,
                    source_passage_id=doc.source_passage_id,
                    strategy=doc.strategy,
                    score=doc.score,
                    is_selected=doc.is_selected or False,
                    snippet=snippet_text
                )
            )

        total_pipeline_time = (time.time() - t_pipeline_start) * 1000 + stt_latency_ms
        latency.total_pipeline_ms = round(total_pipeline_time, 2)

        final_answer = llm_res.answer
        if not post_guard.is_grounded:
            logger.warning(f"[{request_id}] Grounding check warning: {post_guard.violation_reason}")

        logger.info(f"[{request_id}] Pipeline completed in {latency.total_pipeline_ms:.2f}ms (Retrieval: {latency.retrieval_ms:.1f}ms, LLM: {latency.llm_generation_ms:.1f}ms)")

        return RAGResponse(
            request_id=request_id,
            query=query,
            answer=final_answer,
            is_safe=True,
            is_grounded=post_guard.is_grounded,
            violation_reason=post_guard.violation_reason,
            citations=citation_cards,
            retrieved_count=len(reranked_docs),
            latency=latency,
            model_used=llm_res.model,
            used_fallback=(llm_res.model == "local-neural-synthesizer")
        )
