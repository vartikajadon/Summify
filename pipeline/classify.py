import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from llm.client import LLMClient

logger = logging.getLogger(__name__)

VALID_TAXONOMY = {
    "concept",
    "setup/install",
    "api-reference",
    "example",
    "troubleshooting",
    "config",
    "faq",
    "other",
}

CLASSIFIER_PROMPT_TEMPLATE = """You are a technical document classifier.
Classify the following text chunk into EXACTLY ONE of these categories:
- concept: high-level explanations, architecture, principles, or background.
- setup/install: installation steps, dependencies, environment setup, prerequisites.
- api-reference: class signatures, method specs, function parameters, REST endpoints.
- example: code snippets, sample usage walkthroughs, tutorials.
- troubleshooting: error messages, debugging steps, common pitfalls, fixes.
- config: YAML/JSON configuration flags, environment variables, settings.
- faq: frequently asked questions and answers.
- other: generic prose, legal text, license, changelog, unclassified content.

Document Context:
Source File: {source_filename}
Heading Path: {heading_path}

Text Chunk:
\"\"\"
{chunk_text}
\"\"\"

Respond strictly with a JSON object in this exact format:
{{"label": "<category>", "reasoning": "<brief explanation>"}}
"""

@dataclass
class ClassifiedChunk:
    chunk_id: str
    text: str
    source_filename: str
    heading_path: str
    chunk_index: int
    label: str
    confidence: Optional[float] = None
    method: str = "zero-shot"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def parse_label_from_response(response_text: str) -> str:
    """Parses JSON response to extract label; falls back to 'other' if parsing fails or invalid label."""
    try:
        # Extract potential JSON block if response contains markdown formatting
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        label = str(data.get("label", "")).strip().lower()
        if label in VALID_TAXONOMY:
            return label
    except Exception as e:
        logger.debug(f"JSON label parsing failed for response '{response_text}': {e}")

    # Heuristic fallback matching if JSON parsing failed but taxonomy keyword present
    response_lower = response_text.lower()
    for cat in VALID_TAXONOMY:
        if cat in response_lower:
            return cat

    return "other"

def heuristic_classify(heading_path: str, text: str) -> Optional[str]:
    """Fast local rule-based heuristic classifier to save LLM API quota and prevent 429 rate limit errors."""
    h_lower = heading_path.lower()
    t_lower = text[:500].lower()
    
    # 1. Setup / Install
    if any(k in h_lower or k in t_lower for k in ["install", "setup", "quickstart", "getting started", "prerequisite", "requirements.txt", "pip install", "npm install"]):
        return "setup/install"
        
    # 2. Troubleshooting
    if any(k in h_lower or k in t_lower for k in ["troubleshoot", "debugging", "error message", "common pitfall", "issue resolution", "known issues"]):
        return "troubleshooting"
        
    # 3. Configuration
    if any(k in h_lower or k in t_lower for k in ["config", "configuration", "settings", "environment variable", "config.yaml", "config.json"]):
        return "config"
        
    # 4. FAQ
    if any(k in h_lower or k in t_lower for k in ["faq", "frequently asked", "q&a"]):
        return "faq"
        
    # 5. API Reference
    if any(k in h_lower or k in t_lower for k in ["api reference", "endpoint", "class signature", "method spec", "parameters:"]):
        return "api-reference"

    # 6. Example
    if any(k in h_lower or k in t_lower for k in ["code example", "sample code", "tutorial", "usage example"]):
        return "example"
        
    # 7. Concept / Prose Book
    if any(k in h_lower for k in ["concept", "architecture", "overview", "introduction", "background", "principle", "chapter"]):
        return "concept"

    # 8. Book / Literature prose (e.g. Think and Grow Rich, Art of War)
    if len(text) > 300 and not any(c in text for c in ["def ", "class ", "import ", "http://", "https://", "{"]):
        return "concept"

    return None

def classify_chunk_zeroshot(chunk_dict: Dict[str, Any], llm_client: LLMClient) -> ClassifiedChunk:
    """
    Classifies a single chunk using zero-shot LLM prompt with structured output constraints.
    Falls back to 'other' or heuristic on unexpected output or failure.
    """
    chunk_id = chunk_dict.get("chunk_id", "unknown::0")
    text = chunk_dict.get("text", "")
    source_filename = chunk_dict.get("source_filename", "unknown")
    heading_path = chunk_dict.get("heading_path", "Root")
    chunk_index = chunk_dict.get("chunk_index", 0)

    prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
        source_filename=source_filename,
        heading_path=heading_path,
        chunk_text=text[:300]  # Truncated to 300 chars to conserve API token quota
    )

    try:
        response = llm_client.generate(
            prompt,
            system_prompt="You are a JSON document classification system. Respond strictly with JSON.",
            max_tokens=60,
            temperature=0.0
        )
        label = parse_label_from_response(response)
    except Exception as e:
        logger.warning(f"Error during zero-shot classification for chunk {chunk_id}: {e}")
        label = heuristic_classify(heading_path, text) or "other"

    return ClassifiedChunk(
        chunk_id=chunk_id,
        text=text,
        source_filename=source_filename,
        heading_path=heading_path,
        chunk_index=chunk_index,
        label=label,
        confidence=None,
        method="zero-shot"
    )

def batch_classify_headings(
    unique_headings: List[Dict[str, Any]],
    llm_client: LLMClient
) -> Dict[Tuple[str, str], str]:
    """
    Classifies multiple section headings in a single batch LLM API call to maximize speed and eliminate 429 rate limit errors.
    """
    if not unique_headings:
        return {}

    results: Dict[Tuple[str, str], str] = {}
    batch_size = 15
    for i in range(0, len(unique_headings), batch_size):
        batch = unique_headings[i:i + batch_size]
        items_str = []
        for idx, item in enumerate(batch, 1):
            h_path = item.get("heading_path", "Root")
            sample = item.get("text", "")[:150].replace("\n", " ")
            items_str.append(f"{idx}. Heading: {h_path} | Content: {sample}")

        prompt = (
            "Classify each section into EXACTLY ONE category from: "
            "[concept, setup/install, api-reference, example, troubleshooting, config, faq, other].\n\n"
            + "\n".join(items_str) +
            "\n\nRespond strictly with a JSON object mapping line numbers to categories, e.g. {\"1\": \"concept\", \"2\": \"setup/install\"}."
        )

        try:
            resp = llm_client.generate(
                prompt,
                system_prompt="You are a batch document section classifier. Respond strictly with JSON.",
                max_tokens=256,
                temperature=0.0
            )
            text = resp.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)
            for idx, item in enumerate(batch, 1):
                key = (item.get("source_filename", "unknown"), item.get("heading_path", "Root"))
                label = str(parsed.get(str(idx), "")).strip().lower()
                if label in VALID_TAXONOMY:
                    results[key] = label
                else:
                    results[key] = heuristic_classify(item.get("heading_path", "Root"), item.get("text", "")) or "concept"
        except Exception as e:
            logger.warning(f"Batch classification failed: {e}. Using heuristics.")
            for item in batch:
                key = (item.get("source_filename", "unknown"), item.get("heading_path", "Root"))
                results[key] = heuristic_classify(item.get("heading_path", "Root"), item.get("text", "")) or "concept"

    return results

def classify_chunks(chunks: List[Dict[str, Any]], llm_client: Optional[LLMClient] = None, max_workers: int = 1) -> List[ClassifiedChunk]:
    """
    Classifies a batch of chunk dicts into ClassifiedChunk objects.
    Uses heuristic pre-classification + batch LLM classification to eliminate 429 rate limit errors.
    """
    if llm_client is None:
        llm_client = LLMClient()

    if not chunks:
        return []

    if llm_client.is_mock_mode():
        return [classify_chunk_zeroshot(c, llm_client) for c in chunks]

    heading_cache: Dict[Tuple[str, str], str] = {}
    unique_headings: List[Dict[str, Any]] = []
    seen_keys = set()

    for c in chunks:
        key = (c.get("source_filename", "unknown"), c.get("heading_path", "Root"))
        if key not in seen_keys:
            seen_keys.add(key)
            h_res = heuristic_classify(c.get("heading_path", "Root"), c.get("text", ""))
            if h_res:
                heading_cache[key] = h_res
            else:
                unique_headings.append(c)

    # Batch classify remaining unique section headings with a single API call
    if unique_headings:
        batch_results = batch_classify_headings(unique_headings, llm_client)
        heading_cache.update(batch_results)

    result: List[ClassifiedChunk] = []
    for c in chunks:
        key = (c.get("source_filename", "unknown"), c.get("heading_path", "Root"))
        label = heading_cache.get(key) or heuristic_classify(c.get("heading_path", "Root"), c.get("text", "")) or "concept"
        result.append(ClassifiedChunk(
            chunk_id=c.get("chunk_id", "unknown::0"),
            text=c.get("text", ""),
            source_filename=c.get("source_filename", "unknown"),
            heading_path=c.get("heading_path", "Root"),
            chunk_index=c.get("chunk_index", 0),
            label=label,
            confidence=None,
            method="heuristic" if key in heading_cache else "zero-shot"
        ))

    return result
