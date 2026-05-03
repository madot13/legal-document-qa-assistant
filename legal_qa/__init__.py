"""Legal document question answering assistant."""

from legal_qa.assistant import LegalQAAssistant
from legal_qa.types import Answer, DocumentChunk, RetrievalResult

__all__ = ["Answer", "DocumentChunk", "LegalQAAssistant", "RetrievalResult"]
