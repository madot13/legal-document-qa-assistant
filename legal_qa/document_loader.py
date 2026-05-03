from __future__ import annotations

from pathlib import Path


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be read."""


def load_text(path: str | Path) -> str:
    """Extract text from a supported document path."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md", ".text"}:
        return file_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".docx":
        return _load_docx(file_path)

    raise DocumentLoadError(f"Unsupported document type: {suffix or '<none>'}")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError("PDF loading requires pypdf. Install requirements.txt.") from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[Page {index}]\n{text}")
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:
        raise DocumentLoadError("DOCX loading requires python-docx. Install requirements.txt.") from exc

    document = docx.Document(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n\n".join(paragraphs)
