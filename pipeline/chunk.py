import os
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_filename: str
    heading_path: str
    chunk_index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def load_chunk_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("chunking", {})
        except Exception as e:
            logger.warning(f"Could not load chunking config from {config_path}: {e}")
    return {"max_chunk_tokens": 500, "overlap_ratio": 0.15}

def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 chars per token in English text."""
    return max(1, len(text) // 4)

def split_long_text_sliding_window(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Splits a long section of text into overlapping windows when it exceeds max_chars, guaranteeing termination."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Try breaking at last space or newline in the second half of the chunk
            sub = text[start:end]
            last_break = max(sub.rfind(" "), sub.rfind("\n"))
            if last_break > (max_chars // 2):
                end = start + last_break

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        # Guarantee start always advances to prevent infinite loops
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks

def chunk_doc(doc_dict: Dict[str, Any], config_path: str = "config.yaml") -> List[Chunk]:
    """
    Splits an ingested document into semantic chunks based on headings and paragraph boundaries.
    Falls back to sliding window if a single section exceeds max_chunk_tokens.
    """
    cfg = load_chunk_config(config_path)
    max_tokens = cfg.get("max_chunk_tokens", 500)
    max_chars = max_tokens * 4
    overlap_ratio = cfg.get("overlap_ratio", 0.15)
    overlap_chars = int(max_chars * overlap_ratio)

    source_filename = doc_dict.get("source_filename", "unknown")
    full_text = doc_dict.get("text", "")
    headings = doc_dict.get("headings", [])
    status = doc_dict.get("status", "success")

    if status == "error" or not full_text.strip():
        return []

    heading_lookup = {h.get("text"): h for h in headings if h.get("text")}

    # Split full text into lines to track heading changes and section blocks
    lines = full_text.splitlines()
    
    sections = []  # List of tuples: (heading_path_str, section_text)
    current_heading_stack: List[Dict[str, Any]] = []
    current_lines: List[str] = []

    def get_current_heading_path() -> str:
        if not current_heading_stack:
            return "Root"
        return " > ".join(h["text"] for h in current_heading_stack)

    for line in lines:
        stripped = line.strip()
        heading_match = None
        if stripped.startswith("#"):
            parts = stripped.split(maxsplit=1)
            hashes = parts[0]
            if all(c == "#" for c in hashes) and len(parts) > 1:
                heading_match = {"text": parts[1].strip(), "level": len(hashes)}

        if not heading_match and stripped in heading_lookup:
            heading_match = heading_lookup[stripped]

        if heading_match:
            # Flush existing section
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append((get_current_heading_path(), sec_text))
                current_lines = []

            # Update heading hierarchy stack
            level = heading_match["level"]
            while current_heading_stack and current_heading_stack[-1]["level"] >= level:
                current_heading_stack.pop()
            current_heading_stack.append(heading_match)
        else:
            current_lines.append(line)

    # Flush final section
    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append((get_current_heading_path(), sec_text))

    # Convert sections into final Chunks, applying sliding window if needed
    chunks: List[Chunk] = []
    global_chunk_idx = 0

    for heading_path, sec_text in sections:
        if len(sec_text) <= max_chars:
            chunk_id = f"{source_filename}::{global_chunk_idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=sec_text,
                source_filename=source_filename,
                heading_path=heading_path,
                chunk_index=global_chunk_idx
            ))
            global_chunk_idx += 1
        else:
            # Paragraph split first
            paragraphs = sec_text.split("\n\n")
            accumulated_p = []
            accumulated_len = 0

            for p in paragraphs:
                p_strip = p.strip()
                if not p_strip:
                    continue

                if len(p_strip) > max_chars:
                    # Single giant paragraph > max_chars -> apply sliding window
                    if accumulated_p:
                        combined = "\n\n".join(accumulated_p)
                        chunk_id = f"{source_filename}::{global_chunk_idx}"
                        chunks.append(Chunk(
                            chunk_id=chunk_id,
                            text=combined,
                            source_filename=source_filename,
                            heading_path=heading_path,
                            chunk_index=global_chunk_idx
                        ))
                        global_chunk_idx += 1
                        accumulated_p = []
                        accumulated_len = 0

                    sub_chunks = split_long_text_sliding_window(p_strip, max_chars, overlap_chars)
                    for sub in sub_chunks:
                        chunk_id = f"{source_filename}::{global_chunk_idx}"
                        chunks.append(Chunk(
                            chunk_id=chunk_id,
                            text=sub,
                            source_filename=source_filename,
                            heading_path=heading_path,
                            chunk_index=global_chunk_idx
                        ))
                        global_chunk_idx += 1

                elif accumulated_len + len(p_strip) + 2 <= max_chars:
                    accumulated_p.append(p_strip)
                    accumulated_len += len(p_strip) + 2
                else:
                    combined = "\n\n".join(accumulated_p)
                    chunk_id = f"{source_filename}::{global_chunk_idx}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=combined,
                        source_filename=source_filename,
                        heading_path=heading_path,
                        chunk_index=global_chunk_idx
                    ))
                    global_chunk_idx += 1
                    accumulated_p = [p_strip]
                    accumulated_len = len(p_strip)

            if accumulated_p:
                combined = "\n\n".join(accumulated_p)
                chunk_id = f"{source_filename}::{global_chunk_idx}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=combined,
                    source_filename=source_filename,
                    heading_path=heading_path,
                    chunk_index=global_chunk_idx
                ))
                global_chunk_idx += 1

    return chunks

def chunk_files(raw_docs: List[Dict[str, Any]], config_path: str = "config.yaml") -> List[Chunk]:
    """
    Processes a list of raw_text document dictionaries and outputs all semantic chunks.
    """
    all_chunks = []
    for doc_dict in raw_docs:
        chunks = chunk_doc(doc_dict, config_path=config_path)
        all_chunks.extend(chunks)
    return all_chunks
