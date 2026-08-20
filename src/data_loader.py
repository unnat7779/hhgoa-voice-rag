"""
Data loader module for AI4Bharat MSMARCO-XI dataset.
Handles fetching, streaming parquet batches, caching, and preprocessing passages and queries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import DATA_DIR, DEFAULT_LANGUAGE, MAX_PASSAGES_INDEX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE_PASSAGES = DATA_DIR / f"msmarco_{DEFAULT_LANGUAGE}_passages.jsonl"
CACHE_FILE_QUERIES = DATA_DIR / f"msmarco_{DEFAULT_LANGUAGE}_queries.jsonl"


def _get_parquet_file(language: str = DEFAULT_LANGUAGE, split: str = "train") -> Path:
    """
    Ensures the parquet file exists locally or downloads it.
    """
    prefix_map = {
        "hi": "hin", "as": "asm", "bn": "ben", "gu": "guj", "kn": "kan",
        "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
        "sa": "san", "ta": "tam", "tel": "tel", "ur": "urd"
    }
    lang_code = prefix_map.get(language, "hin")
    split_name = "train" if split == "train" else "val"
    parquet_filename = f"{lang_code}{split_name}.parquet"
    local_parquet = DATA_DIR / parquet_filename

    if not local_parquet.exists():
        import requests
        parquet_url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{split}/{parquet_filename}"
        logger.info(f"Downloading parquet from {parquet_url}...")
        resp = requests.get(parquet_url, stream=True)
        resp.raise_for_status()
        with open(local_parquet, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        logger.info(f"Saved {parquet_filename} to {local_parquet}")

    return local_parquet


def stream_raw_records(language: str = DEFAULT_LANGUAGE, split: str = "train", max_records: int = 5000) -> List[Dict[str, Any]]:
    """
    Streams records using PyArrow ParquetFile without loading the entire multi-GB parquet into RAM.
    """
    parquet_path = _get_parquet_file(language=language, split=split)
    logger.info(f"Streaming records from {parquet_path} (target={max_records} records)...")
    
    parquet_file = pq.ParquetFile(str(parquet_path))
    records: List[Dict[str, Any]] = []

    for batch in parquet_file.iter_batches(batch_size=1000):
        batch_dicts = batch.to_pylist()
        records.extend(batch_dicts)
        if len(records) >= max_records:
            records = records[:max_records]
            break

    logger.info(f"Loaded {len(records)} records from parquet.")
    return records


def extract_passages_and_queries(
    records: List[Dict[str, Any]], 
    max_passages: int = MAX_PASSAGES_INDEX
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts deduplicated passages with rich metadata, and a test query set.
    """
    passages: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    seen_passage_texts = set()

    logger.info(f"Processing {len(records)} records for passage extraction (target={max_passages} passages)...")

    for record in tqdm(records, desc="Extracting passages"):
        query_id = record.get("query_id")
        query_text = record.get("query", "")
        eng_query = record.get("Eng_Query", "")
        query_type = record.get("query_type", "unknown")
        answer = record.get("Answer", "")
        eng_answer = record.get("Eng_Answer", "")
        source_lang = record.get("source_lang", "eng_Latn")
        target_lang = record.get("target_lang", "hin_Deva")

        # Save query metadata
        queries.append({
            "query_id": query_id,
            "query": query_text,
            "eng_query": eng_query,
            "query_type": query_type,
            "answer": answer,
            "eng_answer": eng_answer,
            "source_lang": source_lang,
            "target_lang": target_lang
        })

        raw_passages = record.get("passages", {})
        eng_passages_list = []
        trans_passages_list = []
        is_selected_list = []

        if isinstance(raw_passages, dict):
            eng_passages_list = raw_passages.get("English_passages", []) or []
            trans_passages_list = raw_passages.get("Translated_passages", []) or []
            is_selected_list = raw_passages.get("is_selected", []) or []
        elif isinstance(raw_passages, list):
            for p in raw_passages:
                if isinstance(p, dict):
                    eng_passages_list.append(p.get("English_passages", "") or p.get("passage_text", ""))
                    trans_passages_list.append(p.get("Translated_passages", ""))
                    is_selected_list.append(p.get("is_selected", 0))

        for idx, eng_p in enumerate(eng_passages_list):
            if not eng_p or not isinstance(eng_p, str) or len(eng_p.strip()) < 10:
                continue

            cleaned_eng_p = eng_p.strip()
            if cleaned_eng_p in seen_passage_texts:
                continue
            seen_passage_texts.add(cleaned_eng_p)

            is_sel = bool(is_selected_list[idx]) if idx < len(is_selected_list) else False
            trans_p = trans_passages_list[idx] if idx < len(trans_passages_list) else ""

            passage_id = f"p_{query_id}_{idx}"
            passages.append({
                "passage_id": passage_id,
                "text": cleaned_eng_p,
                "translated_text": trans_p,
                "query_id": query_id,
                "query_type": query_type,
                "is_selected": is_sel,
                "source_lang": source_lang,
                "target_lang": target_lang
            })

            if len(passages) >= max_passages:
                break

        if len(passages) >= max_passages:
            break

    logger.info(f"Extracted {len(passages)} unique passages and {len(queries)} query contexts.")
    return passages, queries


def get_or_create_corpus(
    language: str = DEFAULT_LANGUAGE, 
    max_passages: int = MAX_PASSAGES_INDEX,
    force_reload: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns cached passages and queries, or generates and caches them.
    """
    if not force_reload and CACHE_FILE_PASSAGES.exists() and CACHE_FILE_QUERIES.exists():
        logger.info(f"Loading corpus from cache: {CACHE_FILE_PASSAGES}")
        passages = []
        with open(CACHE_FILE_PASSAGES, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    passages.append(json.loads(line))

        queries = []
        with open(CACHE_FILE_QUERIES, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))

        if len(passages) >= min(1000, max_passages):
            return passages[:max_passages], queries

    # If not cached, stream raw records with pyarrow
    needed_records = max(1000, int(max_passages * 1.5))
    raw_records = stream_raw_records(language=language, split="train", max_records=needed_records)
    passages, queries = extract_passages_and_queries(raw_records, max_passages=max_passages)

    # Cache to disk
    logger.info(f"Caching passages to {CACHE_FILE_PASSAGES}...")
    with open(CACHE_FILE_PASSAGES, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Caching queries to {CACHE_FILE_QUERIES}...")
    with open(CACHE_FILE_QUERIES, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return passages, queries


if __name__ == "__main__":
    p, q = get_or_create_corpus(max_passages=2000)
    print(f"Loaded {len(p)} passages and {len(q)} queries.")
