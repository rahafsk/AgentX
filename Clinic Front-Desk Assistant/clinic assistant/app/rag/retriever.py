"""Lexical retrieval over clinic-approved Markdown and text documents.

This module retrieves evidence and source names; answer generation and safe
fallback wording belong to the clinic-knowledge service.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOCUMENTS_DIR = _PROJECT_ROOT / "data" / "clinic_documents"
_SUPPORTED_SUFFIXES = {".md", ".txt"}
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "you",
        "your",
    }
)
_DEMO_DIRECTORY = "demo_clinic_documents"
_DEMO_FILE_PREFIX = "DEMO_ONLY_"


class DocumentLoadError(RuntimeError):
    """Raised when an approved document cannot be read safely."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved evidence chunk with a stable source name and relevance score."""

    text: str
    source: str
    score: float


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    text: str
    source: str
    term_frequencies: Counter[str]
    token_count: int


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (match.casefold() for match in _TOKEN_PATTERN.findall(text))
        if token not in _STOP_WORDS
    )


def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping, word-boundary-aware chunks."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break

        next_start = max(start + 1, end - chunk_overlap)
        while next_start < len(normalized) and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


class ClinicDocumentRetriever:
    """Build a deterministic BM25-style index of approved Markdown/text files.

    The default source is `data/clinic_documents/`. Fictional demo material in
    `data/demo_clinic_documents/` or files prefixed `DEMO_ONLY_` are excluded.
    Recreate the retriever after documents change to refresh the in-memory index.
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        allow_demo_documents: bool = False,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")
        if allow_demo_documents:
            app_environment = (os.getenv("APP_ENV") or "production").strip().lower()
            if app_environment not in {"development", "test"}:
                raise ValueError("Demo documents may only be loaded in development or test.")

        root = Path(documents_dir) if documents_dir is not None else _DEFAULT_DOCUMENTS_DIR
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
        self.documents_dir = root.expanduser().resolve()
        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Clinic documents directory does not exist: {self.documents_dir}"
            )
        if not self.documents_dir.is_dir():
            raise NotADirectoryError(
                f"Clinic documents path is not a directory: {self.documents_dir}"
            )

        self._chunks = self._load_chunks(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            allow_demo_documents=allow_demo_documents,
        )
        self._document_frequency: Counter[str] = Counter()
        for chunk in self._chunks:
            self._document_frequency.update(chunk.term_frequencies.keys())
        self._average_length = (
            sum(chunk.token_count for chunk in self._chunks) / len(self._chunks)
            if self._chunks
            else 0.0
        )

    @property
    def chunk_count(self) -> int:
        """Number of searchable chunks loaded from the approved document directory."""
        return len(self._chunks)

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        """Return the highest-scoring evidence chunks, or an empty list if none match."""
        if not isinstance(query, str):
            raise TypeError("query must be text.")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        query_terms = set(_tokenize(query))
        if not query_terms or not self._chunks:
            return []

        total_chunks = len(self._chunks)
        average_length = self._average_length or 1.0
        ranked: list[RetrievedChunk] = []

        for chunk in self._chunks:
            score = 0.0
            for term in query_terms:
                frequency = chunk.term_frequencies.get(term, 0)
                if not frequency:
                    continue

                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (total_chunks - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_normalizer = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * chunk.token_count / average_length
                )
                score += inverse_frequency * frequency * 2.5 / length_normalizer

            if score > 0:
                ranked.append(RetrievedChunk(text=chunk.text, source=chunk.source, score=score))

        ranked.sort(key=lambda result: (-result.score, result.source))
        return ranked[:top_k]

    def _load_chunks(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        allow_demo_documents: bool,
    ) -> list[_IndexedChunk]:
        chunks: list[_IndexedChunk] = []
        for path in sorted(self.documents_dir.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix.lower() not in _SUPPORTED_SUFFIXES
            ):
                continue

            relative_path = path.relative_to(self.documents_dir)
            if not allow_demo_documents and self._is_demo_path(relative_path):
                continue

            try:
                contents = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                source = relative_path.as_posix()
                raise DocumentLoadError(f"Could not read clinic document: {source}") from exc

            source = relative_path.as_posix()
            for text in _split_text(contents, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                frequencies = Counter(_tokenize(text))
                if frequencies:
                    chunks.append(
                        _IndexedChunk(
                            text=text,
                            source=source,
                            term_frequencies=frequencies,
                            token_count=sum(frequencies.values()),
                        )
                    )
        return chunks

    @staticmethod
    def _is_demo_path(relative_path: Path) -> bool:
        return (
            _DEMO_DIRECTORY in relative_path.parts
            or relative_path.name.startswith(_DEMO_FILE_PREFIX)
            or any(part.startswith(".") for part in relative_path.parts)
        )
