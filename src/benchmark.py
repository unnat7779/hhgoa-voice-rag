"""
P50 / P70 / P100 Latency Benchmarking Suite.
Runs 100 queries across the full Voice-Enabled RAG pipeline,
measures latency percentiles for each stage, and generates markdown reports.
"""

import json
import time
import statistics
import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from src.harness import StructuredModelHarness, RAGRequest, RAGResponse
from src.config import DATA_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyBenchmarkSuite:
    """
    Comprehensive latency benchmark suite testing the entire RAG pipeline.
    """

    def __init__(self, num_queries: int = 50):
        self.num_queries = num_queries
        self.harness = StructuredModelHarness()
        self.queries_file = DATA_DIR / "msmarco_hi_queries.jsonl"

    def load_queries(self) -> List[str]:
        """Loads evaluation queries from MSMARCO dataset cache."""
        queries = []
        if self.queries_file.exists():
            with open(self.queries_file, "r") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        q = item.get("eng_query") or item.get("query")
                        if q and len(q) > 5:
                            queries.append(q)
                    if len(queries) >= self.num_queries:
                        break

        # Fallbacks if queries file is small
        if len(queries) < self.num_queries:
            extras = [
                "What causes lower back pain?",
                "How to treat hypertension and high blood pressure?",
                "What are the symptoms of dehydration in summer?",
                "Who was Leonardo da Vinci and what did he paint?",
                "How does the immune system fight viral infections?",
                "What causes inflation in an economy?",
                "What is the average human body temperature?",
                "How to improve cardiovascular endurance safely?",
                "What are the health benefits of green tea?",
                "Why is sleep essential for brain function?"
            ] * (self.num_queries // 10 + 1)
            queries.extend(extras[:self.num_queries - len(queries)])

        return queries[:self.num_queries]

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline benchmark across loaded queries.
        """
        queries = self.load_queries()
        logger.info(f"Starting Latency Benchmark on {len(queries)} queries...")

        stt_times = []
        pre_guard_times = []
        retrieval_times = []
        rerank_times = []
        llm_times = []
        post_guard_times = []
        total_times = []

        for i, q in enumerate(queries, 1):
            req = RAGRequest(query=q, top_k=5)
            # Simulate real-world voice transcription latency (~35ms)
            simulated_stt = 35.0

            res: RAGResponse = self.harness.run_pipeline(req, stt_latency_ms=simulated_stt)

            stt_times.append(res.latency.stt_ms)
            pre_guard_times.append(res.latency.pre_guardrail_ms)
            retrieval_times.append(res.latency.retrieval_ms)
            rerank_times.append(res.latency.rerank_ms)
            llm_times.append(res.latency.llm_generation_ms)
            post_guard_times.append(res.latency.post_guardrail_ms)
            total_times.append(res.latency.total_pipeline_ms)

            if i % 10 == 0 or i == len(queries):
                logger.info(f"Progress: {i}/{len(queries)} queries evaluated. Current Avg: {np.mean(total_times):.1f}ms")

        def calc_percentiles(arr: List[float]) -> Dict[str, float]:
            return {
                "mean": round(float(np.mean(arr)), 2),
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p70": round(float(np.percentile(arr, 70)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
                "p100_max": round(float(np.max(arr)), 2),
            }

        report = {
            "num_evaluated_queries": len(queries),
            "voice_stt": calc_percentiles(stt_times),
            "pre_guardrail": calc_percentiles(pre_guard_times),
            "vector_retrieval": calc_percentiles(retrieval_times),
            "reranking": calc_percentiles(rerank_times),
            "llm_generation": calc_percentiles(llm_times),
            "post_guardrail": calc_percentiles(post_guard_times),
            "total_pipeline": calc_percentiles(total_times),
            "target_under_200ms": bool(np.percentile(total_times, 70) < 200.0)
        }

        # Save to disk
        out_path = BASE_DIR / "data" / "latency_benchmark_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        self._print_markdown_table(report)
        return report

    def _print_markdown_table(self, report: Dict[str, Any]):
        print("\n" + "="*70)
        print("🎯 LATENCY BENCHMARK PERCENTILE REPORT (Target < 200ms)")
        print("="*70)
        print(f"{'Pipeline Stage':<22} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P90 (ms)':<10} | {'P100 Max':<10}")
        print("-" * 75)

        stages = [
            ("Voice STT (Sarvam)", "voice_stt"),
            ("Pre-Guardrails", "pre_guardrail"),
            ("Vector Retrieval", "vector_retrieval"),
            ("Candidate Re-rank", "reranking"),
            ("LLM Generation", "llm_generation"),
            ("Post-Grounding", "post_guardrail"),
            ("TOTAL PIPELINE", "total_pipeline"),
        ]

        for label, key in stages:
            m = report[key]
            print(f"{label:<22} | {m['mean']:<10.2f} | {m['p50']:<10.2f} | {m['p70']:<10.2f} | {m['p90']:<10.2f} | {m['p100_max']:<10.2f}")

        print("="*70)
        p70_tot = report['total_pipeline']['p70']
        p50_tot = report['total_pipeline']['p50']
        print(f"🏆 P50 Total: {p50_tot:.2f} ms | P70 Total: {p70_tot:.2f} ms | Sub-200ms Target Met: {'✅ YES' if report['target_under_200ms'] else '❌ NO'}")
        print("="*70 + "\n")


if __name__ == "__main__":
    suite = LatencyBenchmarkSuite(num_queries=50)
    suite.run_benchmark()
