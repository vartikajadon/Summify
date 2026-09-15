import os
import pytest
from pathlib import Path
from pipeline.ingest import ingest_files, IngestedDoc
from graph.build_graph import build_graph, LucidocState

@pytest.fixture
def test_fixtures(tmp_path):
    """
    Creates temporary sample fixture files for testing ingestion (.md, .txt, .docx, .pdf, corrupt).
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    # 1. Markdown fixture
    md_file = fixtures_dir / "sample.md"
    md_file.write_text(
        "# Project Overview\n\nLucidoc is an AI document processor.\n\n## Installation\n\nRun pip install -r requirements.txt.",
        encoding="utf-8"
    )

    # 2. Text fixture
    txt_file = fixtures_dir / "sample.txt"
    txt_file.write_text("Simple plain text document content.", encoding="utf-8")

    # 3. DOCX fixture
    docx_file = fixtures_dir / "sample.docx"
    try:
        import docx
        doc = docx.Document()
        doc.add_heading("User Manual", level=1)
        doc.add_paragraph("Welcome to Lucidoc setup guide.")
        doc.add_heading("Configuration", level=2)
        doc.add_paragraph("Edit config.yaml to customize options.")
        doc.save(str(docx_file))
    except Exception as e:
        pytest.skip(f"python-docx unavailable or failed: {e}")

    # 4. PDF fixture
    pdf_file = fixtures_dir / "sample.pdf"
    # Create minimal valid PDF content using pdfplumber / reportlab or simple pdf header
    # Simple binary mock for pdf: we'll create a minimal PDF via standard minimal pdf bytes if possible
    minimal_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kinds [] /Count 1 /Kids [3 0 R]>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>> >> endobj\n"
        b"4 0 obj <</Length 44>> stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello Lucidoc PDF) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000338 00000 n \n"
        b"trailer <</Size 6 /Root 1 0 R>>\n"
        b"startxref\n412\n%%EOF\n"
    )
    pdf_file.write_bytes(minimal_pdf_bytes)

    # 5. Corrupt file fixture
    corrupt_file = fixtures_dir / "corrupt.pdf"
    corrupt_file.write_bytes(b"NOT_A_REAL_PDF_HEADER_CORRUPT_DATA_12345")

    # 6. Unsupported file fixture
    unsupported_file = fixtures_dir / "app.exe"
    unsupported_file.write_bytes(b"\x7fELF_UNSUPPORTED_BINARY")

    return {
        "md": str(md_file),
        "txt": str(txt_file),
        "docx": str(docx_file),
        "pdf": str(pdf_file),
        "corrupt": str(corrupt_file),
        "unsupported": str(unsupported_file),
    }

def test_ingest_markdown(test_fixtures):
    results = ingest_files([test_fixtures["md"]])
    assert len(results) == 1
    doc = results[0]
    assert doc.status == "success"
    assert doc.source_filename == "sample.md"
    assert "Lucidoc is an AI document processor" in doc.text
    assert len(doc.headings) == 2
    assert doc.headings[0] == {"text": "Project Overview", "level": 1}
    assert doc.headings[1] == {"text": "Installation", "level": 2}

def test_ingest_txt(test_fixtures):
    results = ingest_files([test_fixtures["txt"]])
    assert len(results) == 1
    doc = results[0]
    assert doc.status == "success"
    assert doc.source_filename == "sample.txt"
    assert "Simple plain text document content" in doc.text

def test_ingest_docx(test_fixtures):
    results = ingest_files([test_fixtures["docx"]])
    assert len(results) == 1
    doc = results[0]
    assert doc.status == "success"
    assert doc.source_filename == "sample.docx"
    assert "Welcome to Lucidoc setup guide" in doc.text
    assert len(doc.headings) == 2
    assert doc.headings[0] == {"text": "User Manual", "level": 1}
    assert doc.headings[1] == {"text": "Configuration", "level": 2}

def test_ingest_pdf(test_fixtures):
    results = ingest_files([test_fixtures["pdf"]])
    assert len(results) == 1
    doc = results[0]
    assert doc.status == "success"
    assert doc.source_filename == "sample.pdf"
    assert "Hello Lucidoc PDF" in doc.text

def test_ingest_corrupt_and_unsupported_batch(test_fixtures):
    """
    Ensures that bad/corrupt/unsupported files return status='error' and do NOT crash the batch.
    """
    file_batch = [
        test_fixtures["md"],
        test_fixtures["corrupt"],
        test_fixtures["unsupported"],
        test_fixtures["txt"]
    ]
    results = ingest_files(file_batch)
    assert len(results) == 4
    
    # Valid files succeed
    assert results[0].status == "success"
    assert results[3].status == "success"

    # Corrupt & unsupported files return status='error'
    assert results[1].status == "error"
    assert results[1].error_message is not None
    assert results[2].status == "error"

def test_graph_ingest_node_integration(test_fixtures):
    """
    Verifies that LangGraph ingest_node populates raw_text state field.
    """
    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [test_fixtures["md"], test_fixtures["txt"]],
        "session_id": "test_ingest_graph_session"
    }
    result = graph.invoke(state)

    raw_docs = result.get("raw_text", [])
    assert len(raw_docs) == 2
    assert raw_docs[0]["source_filename"] == "sample.md"
    assert raw_docs[1]["source_filename"] == "sample.txt"
    assert raw_docs[0]["status"] == "success"
