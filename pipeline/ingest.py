import os
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class HeadingItem:
    text: str
    level: int

@dataclass
class IngestedDoc:
    source_filename: str
    text: str
    headings: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "success"  # "success" or "error"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def parse_markdown_headings(text: str) -> List[Dict[str, Any]]:
    headings = []
    for line in text.splitlines():
        line_strip = line.strip()
        if line_strip.startswith("#"):
            parts = line_strip.split(maxsplit=1)
            hashes = parts[0]
            if all(c == "#" for c in hashes) and len(parts) > 1:
                level = len(hashes)
                heading_text = parts[1].strip()
                headings.append({"text": heading_text, "level": level})
    return headings

def ingest_markdown_or_txt(file_path: Path) -> IngestedDoc:
    content = ""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode file {file_path.name} with standard encodings.")

    is_md = file_path.suffix.lower() == ".md"
    headings = parse_markdown_headings(content) if is_md else []
    return IngestedDoc(
        source_filename=file_path.name,
        text=content,
        headings=headings,
        status="success"
    )

def ingest_docx(file_path: Path) -> IngestedDoc:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx package is required for docx ingestion.")

    doc = docx.Document(str(file_path))
    full_text = []
    headings = []

    for para in doc.paragraphs:
        text = para.text
        if not text.strip():
            continue
        full_text.append(text)

        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            try:
                level = int(style_name.replace("Heading", "").strip())
            except ValueError:
                level = 1
            headings.append({"text": text.strip(), "level": level})

    return IngestedDoc(
        source_filename=file_path.name,
        text="\n\n".join(full_text),
        headings=headings,
        status="success"
    )

def ingest_pdf(file_path: Path) -> IngestedDoc:
    full_text = []
    headings = []

    # 1. Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
                    try:
                        words = page.extract_words(extra_attrs=["size"])
                        lines: Dict[float, List[Dict[str, Any]]] = {}
                        for w in words:
                            top = round(w["top"], 1)
                            lines.setdefault(top, []).append(w)
                        
                        for top, line_words in lines.items():
                            line_str = " ".join(w["text"] for w in line_words).strip()
                            avg_size = sum(w.get("size", 10) for w in line_words) / len(line_words)
                            if avg_size > 14 and len(line_str) < 100:
                                headings.append({"text": line_str, "level": 1 if avg_size > 18 else 2})
                    except Exception as e:
                        logger.debug(f"Could not extract heading heuristics for PDF page: {e}")
    except Exception as e:
        logger.debug(f"pdfplumber extraction failed, trying pypdf fallback: {e}")
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    full_text.append(t)
        except Exception as e2:
            raise RuntimeError(f"PDF extraction failed with both pdfplumber and pypdf: {e2}")

    extracted_text = "\n\n".join(full_text)
    if not extracted_text.strip():
        logger.warning(f"No extractable text layer found in PDF {file_path.name}.")
        return IngestedDoc(
            source_filename=file_path.name,
            text="",
            headings=[],
            status="error",
            error_message="No extractable text layer found (file may be a scanned PDF image or empty)."
        )

    return IngestedDoc(
        source_filename=file_path.name,
        text=extracted_text,
        headings=headings,
        status="success"
    )

def ingest_single_file(file_path_str: str) -> IngestedDoc:
    path = Path(file_path_str)
    if not path.exists():
        return IngestedDoc(
            source_filename=path.name,
            text="",
            status="error",
            error_message=f"File does not exist: {file_path_str}"
        )

    ext = path.suffix.lower()
    try:
        if ext in [".md", ".txt"]:
            return ingest_markdown_or_txt(path)
        elif ext == ".docx":
            return ingest_docx(path)
        elif ext == ".pdf":
            return ingest_pdf(path)
        else:
            return IngestedDoc(
                source_filename=path.name,
                text="",
                status="error",
                error_message=f"Unsupported file format: {ext}"
            )
    except Exception as e:
        logger.warning(f"Error ingesting file {path.name}: {e}")
        return IngestedDoc(
            source_filename=path.name,
            text="",
            status="error",
            error_message=str(e)
        )

def ingest_files(file_paths: List[str]) -> List[IngestedDoc]:
    """
    Ingests a list of files (.md, .txt, .docx, .pdf), normalizing them to IngestedDoc objects.
    Fault-tolerant: corrupt or unsupported files return status='error' and do not crash the batch.
    """
    results = []
    for fp in file_paths:
        doc = ingest_single_file(fp)
        results.append(doc)
    return results
