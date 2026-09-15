import os
import io
import logging
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)

# Standard taxonomy ordering per PRD Section 3.1 & FR-3.1
CATEGORY_ORDER = [
    "concept",
    "setup/install",
    "config",
    "api-reference",
    "example",
    "troubleshooting",
    "faq",
    "other",
]

CATEGORY_TITLES = {
    "concept": "1. Overview & Key Themes",
    "setup/install": "2. Installation & Setup",
    "config": "3. Configuration & Settings",
    "api-reference": "4. API & Component Reference",
    "example": "5. Examples & Guides",
    "troubleshooting": "6. Troubleshooting & Notes",
    "faq": "7. Frequently Asked Questions",
    "other": "8. Additional Content",
}

def clean_heading_path(hp: str) -> str:
    """Strips out Table of Contents, Index, and TOC boilerplate from heading paths."""
    if not hp or hp.strip().lower() in ["root", "table of contents", "contents", "index", "toc", "table of content"]:
        return "Root"
    parts = [p.strip() for p in hp.split(">")]
    clean_parts = [p for p in parts if p.lower() not in ["root", "table of contents", "contents", "index", "toc", "table of content"]]
    return " > ".join(clean_parts) if clean_parts else "Root"

def is_noisy_chunk(text: str) -> bool:
    """Detects boilerplate noise, empty fragments, SVG graphic code, or pure TOC lines."""
    t = text.strip()
    if not t or len(t) < 5:
        return True
    t_lower = t.lower()
    if any(p in t_lower for p in ["table of contents...", "page 1 of", "page 2 of", "all rights reserved", "copyright ©"]):
        if len(t.split()) < 15:
            return True
    if any(pattern in t_lower for pattern in ["<path d=", "<svg", "fill-rule=", "clip-rule=", "viewbox=", "xmlns="]):
        return True
    return False

SYNTHESIS_SYSTEM_PROMPT = """You are a master document editor and content synthesizer.
Your task is to transform raw notes into a beautifully formatted, structured, and publication-ready summary section.
CRITICAL GUIDELINES:
1. Preserve the EXACT subject matter, genre, and context of the source notes. If the text is a story, narrative, or non-technical file, summarize the plot, characters, and events faithfully.
2. ABSOLUTELY DO NOT invent software architecture, microservices, databases, Kafka/S3/Vault boilerplate, or code snippets if they are not present in the source notes.
3. Group notes logically under clear Markdown subheadings with structured bullet points and key takeaways.
4. Output only the polished Markdown content for this section."""

BATCH_SYNTHESIS_SYSTEM_PROMPT = """You are a master document editor and content synthesizer.
Your task is to transform raw notes across multiple document sections into a beautifully formatted, structured, and publication-ready summary.
CRITICAL GUIDELINES:
1. Format each section under its corresponding section heading.
2. Preserve the EXACT subject matter, genre, and context of the source notes. If the text is a story, narrative, or non-technical file, summarize the plot, characters, and events faithfully.
3. ABSOLUTELY DO NOT invent software architecture, microservices, databases, Kafka/S3/Vault boilerplate, or code snippets if they are not present in the source notes.
4. Organize each section into clear subheadings, structured bullet points, and key takeaways.
5. Output only the polished Markdown content."""

def batch_synthesize_sections(
    category_groups: Dict[str, List[Dict[str, Any]]],
    llm_client: Optional[Any] = None
) -> Dict[str, str]:
    """
    Synthesizes multiple section categories in a single batch LLM API call to maximize speed and eliminate 429 rate limit errors.
    Returns a dict mapping category name -> synthesized markdown content string.
    """
    results: Dict[str, str] = {}
    if not category_groups:
        return results

    if llm_client is None:
        try:
            from llm.client import LLMClient
            llm_client = LLMClient()
        except Exception:
            pass

    is_mock = llm_client and getattr(llm_client, "is_mock_mode", lambda: True)()
    if not llm_client or is_mock:
        for cat, chunks in category_groups.items():
            valid_chunks = [c for c in chunks if not is_noisy_chunk(c.get("text", ""))]
            lines = []
            for c in valid_chunks:
                heading = clean_heading_path(c.get("heading_path", "Root"))
                text = c.get("text", "").strip()
                if heading != "Root":
                    lines.append(f"### {heading}\n{text}")
                else:
                    lines.append(text)
            results[cat] = "\n\n".join(lines)
        return results

    section_prompts = []
    active_cats = []
    for cat in CATEGORY_ORDER:
        chunks = category_groups.get(cat, [])
        valid_chunks = [c for c in chunks if not is_noisy_chunk(c.get("text", ""))]
        if not valid_chunks:
            continue

        active_cats.append(cat)
        display_title = CATEGORY_TITLES.get(cat, cat.title())
        notes = []
        for idx, c in enumerate(valid_chunks[:15], 1):
            clean_h = clean_heading_path(c.get("heading_path", "Root"))
            notes.append(f"Note [{idx}] ({clean_h}): {c.get('text', '').strip()[:1000]}")
        notes_str = "\n\n".join(notes)
        section_prompts.append(f"=== SECTION: {display_title} ({cat}) ===\n{notes_str}")

    if not active_cats:
        return results

    prompt = (
        "Please synthesize the following document sections into a single publication-ready structured document:\n\n"
        + "\n\n".join(section_prompts)
    )

    try:
        synthesized_doc = llm_client.generate(
            prompt,
            system_prompt=BATCH_SYNTHESIS_SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.2
        )

        if synthesized_doc and len(synthesized_doc.strip()) > 50:
            current_cat = None
            current_lines = []
            for line in synthesized_doc.splitlines():
                line_lower = line.lower().strip()
                matched_cat = None
                for cat in active_cats:
                    title = CATEGORY_TITLES.get(cat, cat).lower()
                    if title in line_lower or cat.replace("/", " ") in line_lower:
                        matched_cat = cat
                        break

                if matched_cat:
                    if current_cat and current_lines:
                        results[current_cat] = "\n".join(current_lines).strip()
                    current_cat = matched_cat
                    current_lines = [line]
                else:
                    if current_cat:
                        current_lines.append(line)

            if current_cat and current_lines:
                results[current_cat] = "\n".join(current_lines).strip()
    except Exception as e:
        logger.warning(f"Batch section synthesis failed: {e}. Falling back to formatted raw notes.")

    for cat in active_cats:
        if cat not in results or not results[cat].strip():
            chunks = category_groups.get(cat, [])
            valid_chunks = [c for c in chunks if not is_noisy_chunk(c.get("text", ""))]
            lines = []
            for c in valid_chunks:
                heading = clean_heading_path(c.get("heading_path", "Root"))
                text = c.get("text", "").strip()
                if heading != "Root":
                    lines.append(f"### {heading}\n{text}")
                else:
                    lines.append(text)
            results[cat] = "\n\n".join(lines)

    return results

def synthesize_section_content(
    display_title: str,
    chunks: List[Dict[str, Any]],
    llm_client: Optional[Any] = None
) -> str:
    """Synthesizes raw chunks into a polished, executive technical section using LLM enhancement."""
    valid_chunks = [c for c in chunks if not is_noisy_chunk(c.get("text", ""))]
    if not valid_chunks:
        return ""

    if llm_client is None:
        try:
            from llm.client import LLMClient
            llm_client = LLMClient()
        except Exception:
            pass

    if llm_client and getattr(llm_client, "is_mock_mode", lambda: True)():
        lines = []
        for c in valid_chunks:
            heading = clean_heading_path(c.get("heading_path", "Root"))
            text = c.get("text", "").strip()
            if heading != "Root":
                lines.append(f"### {heading}\n{text}")
            else:
                lines.append(text)
        return "\n\n".join(lines)

    raw_notes_list = []
    for idx, c in enumerate(valid_chunks[:10], 1):
        clean_h = clean_heading_path(c.get("heading_path", "Root"))
        raw_notes_list.append(f"Note [{idx}] ({clean_h}):\n{c.get('text', '').strip()}")
    raw_notes = "\n\n".join(raw_notes_list)

    prompt = f"Section Title: {display_title}\n\nRaw Extracted Notes:\n{raw_notes[:3500]}\n\nPlease generate a polished, comprehensive, publication-ready section for this topic."

    if llm_client:
        try:
            synthesized = llm_client.generate(
                prompt,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                max_tokens=1024,
                temperature=0.2
            )
            if synthesized and len(synthesized.strip()) > 30:
                return synthesized.strip()
        except Exception as e:
            logger.warning(f"Section synthesis failed for '{display_title}': {e}")

    lines = []
    for c in valid_chunks:
        heading = clean_heading_path(c.get("heading_path", "Root"))
        text = c.get("text", "").strip()
        if heading != "Root":
            lines.append(f"### {heading}\n{text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)

def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.85
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicates near-duplicate chunks across files based on text similarity ratio.
    Merges source filenames into 'all_sources'.
    """
    deduped: List[Dict[str, Any]] = []
    merged_count = 0

    for chunk in chunks:
        c_text = chunk.get("text", "").strip()
        c_src = chunk.get("source_filename", "")

        is_dup = False
        for existing in deduped:
            e_text = existing.get("text", "").strip()

            ratio = 0.0
            if c_text == e_text:
                ratio = 1.0
            elif c_text and e_text and abs(len(c_text) - len(e_text)) / max(len(c_text), len(e_text)) < 0.3:
                ratio = SequenceMatcher(None, c_text, e_text).ratio()

            if ratio >= similarity_threshold:
                is_dup = True
                merged_count += 1
                sources = existing.get("all_sources", [])
                if not sources and existing.get("source_filename"):
                    sources = [existing["source_filename"]]
                if c_src and c_src not in sources:
                    sources.append(c_src)
                existing["all_sources"] = sources
                break

        if not is_dup:
            c_copy = dict(chunk)
            if "all_sources" not in c_copy:
                c_copy["all_sources"] = [c_src] if c_src else []
            deduped.append(c_copy)

    return deduped, merged_count

def generate_toc(sections: List[StructuredSection]) -> List[Dict[str, Any]]:
    """Generates Table of Contents dictionary entries for structured document sections."""
    toc = []
    for sec in sections:
        toc.append({
            "category": sec.category,
            "title": sec.display_title,
            "chunk_count": len(sec.chunks)
        })
    return toc

@dataclass
class StructuredSection:
    category: str
    display_title: str
    chunks: List[Dict[str, Any]]
    source_files: List[str]
    synthesized_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StructuredDocument:
    title: str
    toc: List[Dict[str, Any]]
    sections: List[StructuredSection]
    merged_duplicates_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "toc": self.toc,
            "sections": [s.to_dict() for s in self.sections],
            "merged_duplicates_count": self.merged_duplicates_count
        }

def structured_doc_from_dict(d: Any) -> StructuredDocument:
    """Reconstructs a StructuredDocument dataclass instance from a dictionary if needed."""
    if isinstance(d, StructuredDocument):
        return d
    if not d or not isinstance(d, dict):
        return StructuredDocument(title="Empty Documentation", toc=[], sections=[])
    
    sections = []
    for s_dict in d.get("sections", []):
        sections.append(StructuredSection(
            category=s_dict.get("category", "other"),
            display_title=s_dict.get("display_title", "Section"),
            chunks=s_dict.get("chunks", []),
            source_files=s_dict.get("source_files", []),
            synthesized_content=s_dict.get("synthesized_content")
        ))
        
    return StructuredDocument(
        title=d.get("title", "Unified Technical Documentation"),
        toc=d.get("toc", []),
        sections=sections,
        merged_duplicates_count=d.get("merged_duplicates_count", 0)
    )

def assemble_document(
    classified_chunks: List[Dict[str, Any]],
    title: str = "Unified Technical Documentation",
    config_path: str = "config.yaml",
    llm_client: Optional[Any] = None
) -> StructuredDocument:
    """
    Assembles classified chunks into a single structured document grouped by category,
    filtering noise, cleaning headings, deduplicating near-duplicates, and synthesizing sections.
    """
    threshold = 0.85
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                threshold = cfg.get("thresholds", {}).get("dedup_similarity", 0.85)
        except Exception:
            pass

    # Clean headings & filter noisy chunks
    cleaned_chunks = []
    for c in classified_chunks:
        text = c.get("text", "").strip()
        if is_noisy_chunk(text):
            continue
        c_copy = dict(c)
        c_copy["heading_path"] = clean_heading_path(c.get("heading_path", "Root"))
        cleaned_chunks.append(c_copy)

    # 1. Deduplicate
    deduped_chunks, merged_count = deduplicate_chunks(cleaned_chunks, similarity_threshold=threshold)

    # 2. Group by Category
    category_groups: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in CATEGORY_ORDER}
    for chunk in deduped_chunks:
        label = chunk.get("label", "other")
        if label not in category_groups:
            label = "other"
        category_groups[label].append(chunk)

    # 3. Build Structured Sections with Batch Synthesis
    batch_synth = batch_synthesize_sections(category_groups, llm_client=llm_client)

    sections: List[StructuredSection] = []
    for cat in CATEGORY_ORDER:
        chunks_in_cat = category_groups[cat]
        if not chunks_in_cat:
            continue

        display_title = CATEGORY_TITLES.get(cat, cat.title())

        sources_set = []
        for c in chunks_in_cat:
            for s in c.get("all_sources", [c.get("source_filename", "unknown")]):
                if s and s not in sources_set:
                    sources_set.append(s)

        synth_text = batch_synth.get(cat, "")

        sections.append(StructuredSection(
            category=cat,
            display_title=display_title,
            chunks=chunks_in_cat,
            source_files=sources_set,
            synthesized_content=synth_text
        ))

    # 4. Generate TOC
    toc = generate_toc(sections)

    return StructuredDocument(
        title=title,
        toc=toc,
        sections=sections,
        merged_duplicates_count=merged_count
    )

def export_markdown(structured_doc: StructuredDocument) -> str:
    """Renders the StructuredDocument into Markdown format with TOC and source traceability notes."""
    lines = [f"# {structured_doc.title}\n"]

    # Table of Contents
    lines.append("## Table of Contents\n")
    for idx, item in enumerate(structured_doc.toc, 1):
        cat_anchor = item["category"].replace("/", "-").lower()
        lines.append(f"{idx}. [{item['title']}](#{cat_anchor})")
    lines.append("\n---\n")

    # Render Sections
    for sec in structured_doc.sections:
        cat_anchor = sec.category.replace("/", "-").lower()
        lines.append(f"## {sec.display_title} {{#{cat_anchor}}}\n")

        if sec.synthesized_content and sec.synthesized_content.strip():
            lines.append(sec.synthesized_content.strip() + "\n")
        else:
            for chunk in sec.chunks:
                heading = clean_heading_path(chunk.get("heading_path", "Root"))
                if heading and heading != "Root":
                    lines.append(f"### {heading}\n")
                lines.append(chunk.get("text", "").strip() + "\n")

        # Traceability Footer (FR-3.5)
        sources_str = ", ".join(sec.source_files) if sec.source_files else "Unknown"
        lines.append(f"*Source file(s): {sources_str}*\n")
        lines.append("---\n")

    return "\n".join(lines)

def export_docx(structured_doc: StructuredDocument) -> bytes:
    """Exports StructuredDocument to a styled DOCX byte stream using python-docx."""
    try:
        import docx
        from docx.shared import Pt, Inches, RGBColor
    except ImportError:
        raise ImportError("python-docx package is required for export_docx()")

    doc = docx.Document()
    
    # Document Title
    title_p = doc.add_heading(structured_doc.title, level=0)
    
    # Table of Contents Section
    doc.add_heading("Table of Contents", level=1)
    for idx, item in enumerate(structured_doc.toc, 1):
        doc.add_paragraph(f"{idx}. {item['title']} ({item['chunk_count']} sections)")
    doc.add_page_break()

    # Sections
    for sec in structured_doc.sections:
        doc.add_heading(sec.display_title, level=1)

        if sec.synthesized_content and sec.synthesized_content.strip():
            for line in sec.synthesized_content.strip().split("\n"):
                line_str = line.strip()
                if not line_str:
                    continue
                if line_str.startswith("#"):
                    heading_text = line_str.lstrip("#").strip()
                    doc.add_heading(heading_text, level=2)
                else:
                    doc.add_paragraph(line_str)
        else:
            for chunk in sec.chunks:
                heading = clean_heading_path(chunk.get("heading_path", "Root"))
                if heading and heading != "Root":
                    doc.add_heading(heading, level=2)
                
                text = chunk.get("text", "").strip()
                for line in text.split("\n\n"):
                    if line.strip():
                        doc.add_paragraph(line.strip())

        # Source Traceability Footer (FR-3.5)
        sources_str = ", ".join(sec.source_files) if sec.source_files else "Unknown"
        footer_p = doc.add_paragraph()
        run = footer_p.add_run(f"Source file(s): {sources_str}")
        run.font.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(128, 128, 128)
        
        doc.add_paragraph()  # Spacing

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def export_pdf(structured_doc: StructuredDocument) -> bytes:
    """Exports StructuredDocument to PDF bytes (using reportlab or converting markdown)."""
    def clean_p(txt: str) -> str:
        if not txt:
            return ""
        safe = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return safe.replace("\n", "<br/>")

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = styles['Title']
        h1_style = styles['Heading1']
        h2_style = styles['Heading2']
        body_style = styles['BodyText']

        story.append(Paragraph(clean_p(structured_doc.title), title_style))
        story.append(Spacer(1, 12))

        # TOC
        story.append(Paragraph("Table of Contents", h1_style))
        for idx, item in enumerate(structured_doc.toc, 1):
            story.append(Paragraph(clean_p(f"{idx}. {item['title']}"), body_style))
        story.append(Spacer(1, 18))

        # Sections
        for sec in structured_doc.sections:
            story.append(Paragraph(clean_p(sec.display_title), h1_style))
            story.append(Spacer(1, 6))

            if sec.synthesized_content and sec.synthesized_content.strip():
                for line in sec.synthesized_content.strip().split("\n"):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if line_str.startswith("#"):
                        heading_text = line_str.lstrip("#").strip()
                        story.append(Paragraph(clean_p(heading_text), h2_style))
                    else:
                        story.append(Paragraph(clean_p(line_str), body_style))
                    story.append(Spacer(1, 4))
            else:
                for chunk in sec.chunks:
                    heading = clean_heading_path(chunk.get("heading_path", ""))
                    if heading and heading != "Root":
                        story.append(Paragraph(clean_p(heading), h2_style))
                    
                    text = chunk.get("text", "")
                    story.append(Paragraph(clean_p(text), body_style))
                    story.append(Spacer(1, 6))

            sources_str = ", ".join(sec.source_files) if sec.source_files else "Unknown"
            story.append(Paragraph(f"<i>Source file(s): {clean_p(sources_str)}</i>", body_style))
            story.append(Spacer(1, 12))

        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Reportlab PDF generation failed ({e}); falling back to clean text PDF byte stream.")
        md_text = export_markdown(structured_doc)
        buffer = io.BytesIO()
        buffer.write(f"%PDF-1.4\n% Summify Export PDF\n{md_text}".encode("utf-8"))
        return buffer.getvalue()
