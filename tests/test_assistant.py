from __future__ import annotations

import unittest

from legal_qa.assistant import LegalQAAssistant


class LegalQAAssistantTest(unittest.TestCase):
    def test_answers_with_evidence(self) -> None:
        text = """
        1. Termination.
        Either party may terminate this Agreement upon thirty days written notice.

        2. Confidentiality.
        The confidentiality obligations survive termination for three years after disclosure.
        """
        assistant = LegalQAAssistant.from_text(text, source="test")

        result = assistant.ask("Do confidentiality obligations survive termination?")

        self.assertTrue(result.found)
        self.assertIn("Yes.", result.answer)
        self.assertIn("three years", result.evidence)

    def test_returns_not_found_for_unrelated_question(self) -> None:
        assistant = LegalQAAssistant.from_text("This agreement is governed by New York law.", source="test")

        result = assistant.ask("What is the insurance coverage limit?")

        self.assertFalse(result.found)
        self.assertIn("Not found", result.answer)


if __name__ == "__main__":
    unittest.main()
