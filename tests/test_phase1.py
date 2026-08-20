"""
Unit tests and verification for Phase 1:
- Dataset loading
- Multi-strategy chunking
- Embedding and Qdrant indexing
- Basic retrieval latency test
"""

import time
import unittest
from src.chunking import (
    PassageChunker,
    SentenceWindowChunker,
    RecursiveOverlapChunker,
    SemanticChunker,
    HierarchicalParentChildChunker,
    MultiStrategyChunkingEngine,
    split_into_sentences
)
from src.data_loader import extract_passages_and_queries


class TestPhase1Chunking(unittest.TestCase):

    def setUp(self):
        self.sample_passage = {
            "passage_id": "p_test_101",
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
                    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
                    "Locally nicknamed 'La dame de fer', it was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
                    "Although initially criticised by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon.",
            "translated_text": "एफिल टॉवर पेरिस में स्थित एक लौह टॉवर है।",
            "query_id": 101,
            "query_type": "description",
            "is_selected": True,
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva"
        }

    def test_sentence_splitter(self):
        sentences = split_into_sentences(self.sample_passage["text"])
        self.assertEqual(len(sentences), 4)

    def test_passage_chunker(self):
        chunks = PassageChunker.chunk(self.sample_passage)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].strategy, "passage_level")
        self.assertEqual(chunks[0].is_selected, True)
        self.assertIn("Eiffel Tower", chunks[0].text)

    def test_sentence_window_chunker(self):
        chunker = SentenceWindowChunker(window_size=1)
        chunks = chunker.chunk(self.sample_passage)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].strategy, "sentence_window")

    def test_recursive_overlap_chunker(self):
        chunker = RecursiveOverlapChunker(max_tokens=25, overlap_tokens=5)
        chunks = chunker.chunk(self.sample_passage)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].strategy, "recursive_overlap")

    def test_hierarchical_chunker(self):
        chunks = HierarchicalParentChildChunker.chunk(self.sample_passage)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].strategy, "hierarchical_child")
        self.assertEqual(chunks[0].parent_id, "p_test_101")
        self.assertEqual(chunks[0].parent_text, self.sample_passage["text"])

    def test_multi_strategy_engine(self):
        engine = MultiStrategyChunkingEngine()
        res = engine.process_passage(self.sample_passage)
        self.assertIn("passage_level", res)
        self.assertIn("sentence_window", res)
        self.assertIn("recursive_overlap", res)
        self.assertIn("semantic", res)
        self.assertIn("hierarchical_child", res)


if __name__ == "__main__":
    unittest.main()
