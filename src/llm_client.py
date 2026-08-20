"""
LLM Client integration with Groq Cloud (Llama 3.3 / Llama 3.1) and high-speed fallback generation.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Generator
from pydantic import BaseModel
from src.retriever import RetrievedDocument
from src.config import GROQ_API_KEY, GROQ_MODEL_FAST, GROQ_MODEL_SMART

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """You are an ultra-fast, accurate AI Voice Assistant for Retrieval-Augmented Generation (RAG).
Your goal is to answer the user's question accurately using ONLY the provided retrieved context passages.

Guidelines:
1. Grounding: Answer strictly using facts present in the retrieved context. Do NOT extrapolate or hallucinate.
2. Tone: Be concise, clear, and direct — optimal for voice playback.
3. Citations: When referencing facts, attribute them with snippet identifiers like [Doc-1], [Doc-2].
4. Language: If the user asks in Hindi, answer in clear Hindi (or Hinglish if appropriate). If asked in English, answer in English.
5. If the context does not contain enough information to answer the question, state: "Based on the provided documents, I do not have enough information to answer this accurately."

Retrieved Context:
{context_blocks}
"""

class LLMResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float
    token_count: int
    finish_reason: str
    grounded_citations: List[str]


class GroqLLMClient:
    """
    Client for Groq Cloud LLM API with ultrafast inference speed (<100ms TTFT).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = GROQ_MODEL_FAST
    ):
        self.api_key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        self.default_model = default_model
        self.client = None
        
        if self.api_key and self.api_key.strip():
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq client initialized with model: {default_model}")
            except Exception as e:
                logger.warning(f"Failed initializing Groq client: {e}. Falling back to mock generator.")
        else:
            logger.info("No GROQ_API_KEY found. Running in high-performance local fallback generator mode.")

    def format_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Formats retrieved documents into numbered context blocks for prompt injection.
        """
        blocks = []
        for i, doc in enumerate(documents, 1):
            source_tag = f"[Doc-{i}] (Strategy: {doc.strategy}, PassageID: {doc.source_passage_id})"
            blocks.append(f"{source_tag}\n{doc.text.strip()}\n")
        return "\n".join(blocks)

    def generate_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
        model: Optional[str] = None,
        max_tokens: int = 250,
        temperature: float = 0.2
    ) -> LLMResponse:
        """
        Generates grounded answer from retrieved context.
        """
        t0 = time.time()
        context_str = self.format_context(documents)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(context_blocks=context_str)
        target_model = model or self.default_model

        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = completion.choices[0].message.content.strip()
                finish_reason = completion.choices[0].finish_reason or "stop"
                elapsed_ms = (time.time() - t0) * 1000
                tokens = completion.usage.total_tokens if completion.usage else len(answer.split())

                # Extract citations
                citations = [f"Doc-{i+1}" for i, d in enumerate(documents) if f"Doc-{i+1}" in answer or f"[{i+1}]" in answer]

                return LLMResponse(
                    answer=answer,
                    model=target_model,
                    latency_ms=elapsed_ms,
                    token_count=tokens,
                    finish_reason=finish_reason,
                    grounded_citations=citations
                )
            except Exception as e:
                logger.error(f"Groq API error: {e}. Utilizing synthesized local fallback.")

        # Local deterministic synthesis fallback when API key is unavailable or during offline testing
        return self._synthesize_fallback_answer(query, documents, t0)

    def _synthesize_fallback_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
        start_time: float
    ) -> LLMResponse:
        """
        Deterministic local synthesis engine extracting key facts from top retrieved contexts.
        """
        if not documents:
            ans = "I could not find relevant information in the corpus to answer your question."
            citations = []
        else:
            top_doc = documents[0]
            # Form clean concise summary of top retrieved text
            sentences = [s.strip() for s in top_doc.text.split(".") if len(s.strip()) > 10]
            summary = ". ".join(sentences[:2]) + "." if sentences else top_doc.text[:200]
            ans = f"{summary} [Doc-1]"
            citations = ["Doc-1"]

        elapsed_ms = (time.time() - start_time) * 1000
        return LLMResponse(
            answer=ans,
            model="local-neural-synthesizer",
            latency_ms=elapsed_ms,
            token_count=len(ans.split()),
            finish_reason="stop",
            grounded_citations=citations
        )
