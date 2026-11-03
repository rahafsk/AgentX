"""Unit tests for clinic-approved document retrieval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import app.rag.retriever as rag_retriever
from app.rag import ClinicDocumentRetriever
from app.rag.retriever import _split_text


def test_retrieves_relevant_text_with_source_name(tmp_path: Path) -> None:
    (tmp_path / "clinic-hours.md").write_text(
        "The clinic's weekday opening hours are Monday through Friday from 9 AM to 5 PM.",
        encoding="utf-8",
    )
    (tmp_path / "contact.txt").write_text(
        "Contact the front desk by email for general questions.",
        encoding="utf-8",
    )
    retriever = ClinicDocumentRetriever(tmp_path)

    results = retriever.retrieve("What are the weekday opening hours?")

    assert results
    assert results[0].source == "clinic-hours.md"
    assert "9 AM to 5 PM" in results[0].text
    assert results[0].score > 0


def test_returns_no_evidence_when_query_has_no_match(tmp_path: Path) -> None:
    (tmp_path / "clinic-hours.md").write_text(
        "The clinic opens on weekdays.",
        encoding="utf-8",
    )
    retriever = ClinicDocumentRetriever(tmp_path)

    assert retriever.retrieve("quantum spacecraft purple") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("the and is") == []


def test_chunks_long_documents_with_configured_size_and_overlap(tmp_path: Path) -> None:
    (tmp_path / "long-policy.md").write_text(
        "appointment policy details " * 40,
        encoding="utf-8",
    )
    retriever = ClinicDocumentRetriever(tmp_path, chunk_size=80, chunk_overlap=10)

    assert retriever.chunk_count > 1
    results = retriever.retrieve("appointment policy", top_k=10)
    assert results
    assert all(len(result.text) <= 80 for result in results)
    assert all(result.source == "long-policy.md" for result in results)


def test_chunker_preserves_configured_overlap_when_progress_allows() -> None:
    chunks = _split_text("0123456789" * 40, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_large_chunk_overlap_is_capped_to_avoid_redundant_chunks() -> None:
    chunks = _split_text("x" * 10_000, chunk_size=80, chunk_overlap=79)

    assert len(chunks) < 600
    assert chunks[0][-60:] == chunks[1][:60]


def test_ignores_unsupported_hidden_and_demo_files(tmp_path: Path) -> None:
    (tmp_path / "approved.md").write_text(
        "Approved clinic hours are listed here.", encoding="utf-8"
    )
    (tmp_path / "notes.pdf").write_text("unparsed PDF data", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text(
        "Approved clinic hours are hidden content.", encoding="utf-8"
    )
    (tmp_path / "DEMO_ONLY_hours.md").write_text("fictional demo hours", encoding="utf-8")
    demo_directory = tmp_path / "demo_clinic_documents"
    demo_directory.mkdir()
    (demo_directory / "hours.md").write_text("more fictional hours", encoding="utf-8")
    retriever = ClinicDocumentRetriever(tmp_path)

    results = retriever.retrieve("approved clinic hours")

    assert [result.source for result in results] == ["approved.md"]


@pytest.mark.parametrize("use_symlink", [False, True])
def test_demo_documents_directory_is_excluded_without_opt_in(
    tmp_path: Path,
    use_symlink: bool,
) -> None:
    demo_directory = tmp_path / "demo_clinic_documents"
    demo_directory.mkdir()
    (demo_directory / "hours.md").write_text("Demo clinic hours are 9 to 5.", encoding="utf-8")

    documents_dir = demo_directory
    if use_symlink:
        documents_dir = tmp_path / "documents-link"
        try:
            documents_dir.symlink_to(demo_directory, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Symlink creation is unavailable: {exc}")

    retriever = ClinicDocumentRetriever(documents_dir)

    assert retriever.retrieve("demo clinic hours") == []


def test_rejects_demo_documents_outside_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "DEMO_ONLY_hours.md").write_text("fictional hours", encoding="utf-8")
    monkeypatch.setattr(
        rag_retriever, "get_settings", lambda: SimpleNamespace(app_env="production")
    )

    with pytest.raises(ValueError, match="development or test"):
        ClinicDocumentRetriever(tmp_path, allow_demo_documents=True)


def test_demo_documents_can_be_enabled_in_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "DEMO_ONLY_hours.md").write_text("fictional demo clinic hours", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("fictional demo clinic hours hidden", encoding="utf-8")
    monkeypatch.setattr(
        rag_retriever, "get_settings", lambda: SimpleNamespace(app_env="development")
    )
    retriever = ClinicDocumentRetriever(tmp_path, allow_demo_documents=True)

    results = retriever.retrieve("demo clinic hours")

    assert results
    assert [result.source for result in results] == ["DEMO_ONLY_hours.md"]


def test_rejects_invalid_chunk_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        ClinicDocumentRetriever(tmp_path, chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        ClinicDocumentRetriever(tmp_path, chunk_size=50, chunk_overlap=50)


def test_requires_documents_directory(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ClinicDocumentRetriever(missing_directory)
