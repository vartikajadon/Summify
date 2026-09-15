import pytest
from pipeline.structure import assemble_document, deduplicate_chunks, export_markdown, export_docx, export_pdf, StructuredDocument
from graph.build_graph import build_graph, LucidocState

def test_assembly_taxonomy_ordering():
    chunks = [
        {"chunk_id": "1", "text": "Unclassified notes", "source_filename": "f1.md", "heading_path": "Root", "chunk_index": 0, "label": "other"},
        {"chunk_id": "2", "text": "System Architecture Overview", "source_filename": "f2.md", "heading_path": "Arch", "chunk_index": 0, "label": "concept"},
        {"chunk_id": "3", "text": "Run pip install -r requirements.txt", "source_filename": "f3.md", "heading_path": "Install", "chunk_index": 0, "label": "setup/install"},
        {"chunk_id": "4", "text": "def compute(x: int) -> int", "source_filename": "f4.md", "heading_path": "API", "chunk_index": 0, "label": "api-reference"}
    ]

    doc = assemble_document(chunks, title="Test Spec Document")

    categories_in_doc = [sec.category for sec in doc.sections]
    assert categories_in_doc == ["concept", "setup/install", "api-reference", "other"]
    assert len(doc.toc) == 4

def test_deduplication_across_files():
    duplicate_text = "This is a duplicate overview section appearing in multiple files."
    chunks = [
        {"chunk_id": "a1", "text": duplicate_text, "source_filename": "file_a.md", "heading_path": "Intro", "chunk_index": 0, "label": "concept"},
        {"chunk_id": "b1", "text": duplicate_text, "source_filename": "file_b.docx", "heading_path": "Overview", "chunk_index": 0, "label": "concept"},
        {"chunk_id": "c1", "text": "Unique text content here.", "source_filename": "file_c.pdf", "heading_path": "Details", "chunk_index": 0, "label": "concept"}
    ]

    deduped, count = deduplicate_chunks(chunks)
    assert len(deduped) == 2
    assert count == 1

    # Check sources merged
    first_chunk = deduped[0]
    assert set(first_chunk["all_sources"]) == {"file_a.md", "file_b.docx"}

def test_source_traceability_footer():
    chunks = [
        {"chunk_id": "1", "text": "Installation text", "source_filename": "setup.md", "heading_path": "Setup", "chunk_index": 0, "label": "setup/install"}
    ]
    doc = assemble_document(chunks)
    md_output = export_markdown(doc)

    assert "Source file(s): setup.md" in md_output
    assert "## 2. Installation & Setup" in md_output

def test_multi_format_exports():
    chunks = [
        {"chunk_id": "1", "text": "Sample text for export test", "source_filename": "guide.md", "heading_path": "Guide", "chunk_index": 0, "label": "concept"}
    ]
    doc = assemble_document(chunks, title="Export Test Doc")

    md_bytes = export_markdown(doc)
    assert isinstance(md_bytes, str)
    assert len(md_bytes) > 0

    docx_bytes = export_docx(doc)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 500  # Valid binary DOCX file header

    pdf_bytes = export_pdf(doc)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 100

def test_graph_structure_export_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")

    md_path = tmp_path / "struct_test.md"
    md_path.write_text("# Project Concept\n\nHigh level concept explanation.\n\n## Install\n\nRun setup.", encoding="utf-8")

    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [str(md_path)],
        "session_id": "structure_graph_session"
    }

    result = graph.invoke(state)

    structured = result.get("structured_document", {})
    assert "sections" in structured
    assert "toc" in structured
    assert len(structured["sections"]) >= 1
