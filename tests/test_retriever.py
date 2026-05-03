from __future__ import annotations

import unittest

from legal_qa.chunking import chunk_text
from legal_qa.retriever import BM25Retriever


class BM25RetrieverTest(unittest.TestCase):
    def test_retrieves_assignment_clause(self) -> None:
        text = """
        1. Assignment.
        Neither party may assign this Agreement without the prior written consent of the other party.

        2. Governing Law.
        This Agreement is governed by the laws of New York.
        """
        chunks = chunk_text(text, source="test")
        retriever = BM25Retriever(chunks)

        results = retriever.search("Can a party assign the agreement?", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn("assign", results[0].chunk.text.lower())


if __name__ == "__main__":
    unittest.main()
