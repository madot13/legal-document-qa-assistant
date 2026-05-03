from __future__ import annotations

import re
from collections.abc import Iterable

from legal_qa.tokenization import sentence_split, tokenize
from legal_qa.types import DocumentChunk


SECTION_PATTERN = re.compile(r"^\s*(?:section\s+)?(\d+(?:\.\d+)*\.?|[A-Z][A-Z\s]{4,})\s*[:.-]?\s+", re.I)


def chunk_text(
    text: str,
    *,
    source: str,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[DocumentChunk]:
    """Split a document into searchable chunks while preserving source offsets."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    paragraphs = _paragraphs_with_offsets(text)
    chunks: list[DocumentChunk] = []
    section_title = ""

    for paragraph, start, end in paragraphs:
        maybe_title = _section_title(paragraph)
        if maybe_title:
            section_title = maybe_title

        paragraph_tokens = tokenize(paragraph, keep_stopwords=True)
        if len(paragraph_tokens) <= max_tokens:
            chunks.append(
                _make_chunk(
                    source=source,
                    index=len(chunks),
                    text=paragraph,
                    start=start,
                    end=end,
                    section=section_title,
                )
            )
            continue

        for text_part, part_start, part_end in _split_large_paragraph(
            paragraph,
            base_start=start,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ):
            chunks.append(
                _make_chunk(
                    source=source,
                    index=len(chunks),
                    text=text_part,
                    start=part_start,
                    end=part_end,
                    section=section_title,
                )
            )

    return chunks


def chunk_many(items: Iterable[tuple[str, str]], **kwargs: object) -> list[DocumentChunk]:
    """Chunk multiple `(source, text)` pairs into one chunk list."""

    all_chunks: list[DocumentChunk] = []
    for source, text in items:
        all_chunks.extend(chunk_text(text, source=source, **kwargs))
    return all_chunks


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    paragraphs: list[tuple[str, int, int]] = []

    block_lines: list[str] = []
    block_start = 0
    cursor = 0

    for line in text.splitlines(keepends=True):
        if line.strip():
            if not block_lines:
                block_start = cursor
            block_lines.append(line)
        elif block_lines:
            paragraphs.append(_finalize_paragraph(block_lines, block_start))
            block_lines = []
        cursor += len(line)

    if block_lines:
        paragraphs.append(_finalize_paragraph(block_lines, block_start))

    return paragraphs


def _finalize_paragraph(lines: list[str], block_start: int) -> tuple[str, int, int]:
    raw = "".join(lines)
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    paragraph = re.sub(r"[ \t]+", " ", raw.strip())
    return paragraph, block_start + leading, block_start + len(raw) - trailing


def _split_large_paragraph(
    paragraph: str,
    *,
    base_start: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    sentences = sentence_split(paragraph)
    if not sentences:
        return []

    parts: list[tuple[str, int, int]] = []
    window: list[str] = []
    window_token_count = 0
    local_cursor = 0
    window_start = 0

    for sentence in sentences:
        sentence_start = paragraph.find(sentence, local_cursor)
        sentence_end = sentence_start + len(sentence)
        local_cursor = sentence_end
        sentence_token_count = len(tokenize(sentence, keep_stopwords=True))

        if sentence_token_count > max_tokens:
            if window:
                parts.append(_window_to_part(paragraph, window, window_start, base_start))
                window = []
                window_token_count = 0
            parts.extend(_split_by_words(sentence, base_start + sentence_start, max_tokens, overlap_tokens))
            continue

        if window and window_token_count + sentence_token_count > max_tokens:
            parts.append(_window_to_part(paragraph, window, window_start, base_start))
            overlap = _tail_by_token_budget(window, overlap_tokens)
            window = overlap + [sentence]
            window_start = paragraph.find(window[0])
            window_token_count = sum(len(tokenize(item, keep_stopwords=True)) for item in window)
            continue

        if not window:
            window_start = sentence_start
        window.append(sentence)
        window_token_count += sentence_token_count

    if window:
        parts.append(_window_to_part(paragraph, window, window_start, base_start))

    return parts


def _split_by_words(
    sentence: str,
    base_start: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    words = sentence.split()
    step = max(1, max_tokens - overlap_tokens)
    parts = []
    for index in range(0, len(words), step):
        text_part = " ".join(words[index : index + max_tokens])
        start_offset = sentence.find(text_part.split()[0]) if text_part else 0
        parts.append((text_part, base_start + start_offset, base_start + start_offset + len(text_part)))
        if index + max_tokens >= len(words):
            break
    return parts


def _tail_by_token_budget(sentences: list[str], budget: int) -> list[str]:
    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        count = len(tokenize(sentence, keep_stopwords=True))
        if selected and total + count > budget:
            break
        if count > budget:
            break
        selected.insert(0, sentence)
        total += count
    return selected


def _window_to_part(paragraph: str, sentences: list[str], local_start: int, base_start: int) -> tuple[str, int, int]:
    text_part = " ".join(sentences).strip()
    actual_start = paragraph.find(sentences[0], local_start)
    actual_end = paragraph.find(sentences[-1], actual_start) + len(sentences[-1])
    return text_part, base_start + actual_start, base_start + actual_end


def _section_title(paragraph: str) -> str:
    first_line = paragraph.splitlines()[0].strip()
    if len(first_line) > 120:
        return ""
    match = SECTION_PATTERN.match(first_line)
    return first_line if match else ""


def _make_chunk(
    *,
    source: str,
    index: int,
    text: str,
    start: int,
    end: int,
    section: str,
) -> DocumentChunk:
    return DocumentChunk(
        id=f"{source}#{index}",
        text=text,
        source=source,
        start_char=start,
        end_char=end,
        metadata={"section": section} if section else {},
    )
