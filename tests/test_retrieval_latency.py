"""
Retrieval verification and vector search latency test for Phase 1.
Benchmarks Qdrant vector retrieval speed against the indexed MS MARCO corpus.
"""

import time
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.config import (
    QDRANT_PATH,
    EMBEDDING_MODEL_NAME,
    COLLECTION_SEMANTIC,
    COLLECTION_PASSAGE,
    COLLECTION_SENTENCE,
    COLLECTION_HIERARCHICAL,
    DATA_DIR
)

def benchmark_vector_retrieval():
    print("=== Phase 1: Vector DB & Retrieval Latency Benchmark ===")
    
    # 1. Load embedder and Qdrant
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    client = QdrantClient(path=QDRANT_PATH)

    # 2. Check collections
    collections = client.get_collections().collections
    print(f"Active Qdrant Collections: {[c.name for c in collections]}")

    # 3. Load sample queries
    queries_file = DATA_DIR / "msmarco_hi_queries.jsonl"
    test_queries = []
    if queries_file.exists():
        with open(queries_file, "r") as f:
            for i, line in enumerate(f):
                if line.strip():
                    test_queries.append(json.loads(line))
                if len(test_queries) >= 20:
                    break

    if not test_queries:
        test_queries = [
            {"query": "मौसम क्या है?", "eng_query": "what is weather"},
            {"query": "पेरिस की राजधानी क्या है?", "eng_query": "what is the capital of France"},
            {"query": "लौह टॉवर किसने बनाया?", "eng_query": "who built the Eiffel Tower"}
        ]

    print(f"\nRunning benchmark on {len(test_queries)} sample queries...")
    latencies = []

    for item in test_queries:
        query_text = item.get("eng_query") or item.get("query")
        
        t0 = time.time()
        
        # Step A: Query embedding
        q_emb = embedder.encode(query_text, normalize_embeddings=True).tolist()
        t_embed = (time.time() - t0) * 1000

        # Step B: Parallel multi-collection vector search
        t_search_start = time.time()
        
        # Query semantic collection
        res_sem = client.query_points(
            collection_name=COLLECTION_SEMANTIC,
            query=q_emb,
            limit=3
        ).points
        
        # Query passage collection
        res_pass = client.query_points(
            collection_name=COLLECTION_PASSAGE,
            query=q_emb,
            limit=3
        ).points
        
        # Query sentence collection
        res_sent = client.query_points(
            collection_name=COLLECTION_SENTENCE,
            query=q_emb,
            limit=3
        ).points

        # Query hierarchical collection
        res_hier = client.query_points(
            collection_name=COLLECTION_HIERARCHICAL,
            query=q_emb,
            limit=3
        ).points

        t_search = (time.time() - t_search_start) * 1000
        t_total = (time.time() - t0) * 1000
        latencies.append({"embed_ms": t_embed, "search_ms": t_search, "total_ms": t_total})

    # Summary metrics
    avg_embed = sum(x["embed_ms"] for x in latencies) / len(latencies)
    avg_search = sum(x["search_ms"] for x in latencies) / len(latencies)
    avg_total = sum(x["total_ms"] for x in latencies) / len(latencies)

    print("\n" + "="*50)
    print(">>> LATENCY BENCHMARK RESULTS (Across 4 Collections) <<<")
    print(f"Average Query Embedding Latency: {avg_embed:.2f} ms")
    print(f"Average 4-Collection Vector Search: {avg_search:.2f} ms")
    print(f"Total Retrieval (Embed + Search):   {avg_total:.2f} ms")
    print("="*50)

    # Inspect top hit
    top_hit = res_sem[0] if res_sem else None
    if top_hit:
        print("\nSample Retrieved Hit from Qdrant:")
        print(f"- Score: {top_hit.score:.4f}")
        print(f"- Strategy: {top_hit.payload.get('strategy')}")
        print(f"- Text snippet: {top_hit.payload.get('text')[:120]}...")
        print(f"- Parent Passage ID: {top_hit.payload.get('parent_id')}")
        print(f"- Query Type: {top_hit.payload.get('query_type')}")
        print(f"- Is Selected: {top_hit.payload.get('is_selected')}")

    assert avg_total < 50.0, f"Retrieval latency {avg_total:.2f}ms exceeds target budget!"
    print("\n✅ Verification Successful: Vector DB retrieval operates well within sub-50ms budget!")

if __name__ == "__main__":
    benchmark_vector_retrieval()
