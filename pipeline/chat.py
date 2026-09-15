import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from llm.client import LLMClient
from pipeline.index import query_index

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are Summify, an expert AI assistant answering questions about uploaded user documentation.
Your instructions:
1. Primary Goal: Provide clear, comprehensive, detailed, and helpful answers to user questions using the provided Document Context snippets.
2. Context Integration: Incorporate relevant facts, components, props, guidelines, and code snippets from the Document Context, citing sources as [Source: filename > Heading].
3. Synthesis & Guidance: If the user asks general conceptual or procedural questions (e.g., "what is X", "how to create Y") and the Document Context contains related components, props, dynamic expressions, or usage guides, synthesize those details to explain the concept step-by-step. Supplement the document facts with clear, accurate technical domain knowledge.
4. Helpful Attitude: Always attempt to give a constructive, structured response. Only state that information is missing if the document context has absolutely zero relevance to the user's question.
"""

CHAT_USER_PROMPT_TEMPLATE = """Document Context:
{context_text}

Chat History:
{history_text}

User Question: {question}

Please provide a clear, factual answer with source citations matching [Source: filename > Heading].
"""

@dataclass
class DraftAnswer:
    answer_text: str
    citations: List[Dict[str, str]]
    retrieved_context: List[Dict[str, Any]]
    question: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def format_context_for_prompt(chunks: List[Dict[str, Any]], max_chars_per_chunk: int = 600) -> str:
    if not chunks:
        return "No relevant context found in uploaded documents."

    formatted = []
    total_chars = 0
    for idx, c in enumerate(chunks[:4], 1):
        source = c.get("source_filename", "unknown")
        heading = c.get("heading_path", "Root")
        text = c.get("text", "").strip()[:max_chars_per_chunk]
        if total_chars + len(text) > 2500:
            break
        formatted.append(f"Snippet [{idx}] (Source: {source} > {heading}):\n{text}\n")
        total_chars += len(text)
    return "\n".join(formatted) if formatted else "No relevant context found in uploaded documents."

def format_history_for_prompt(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return "None"
    formatted = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)

def extract_citations_from_context(chunks: List[Dict[str, Any]], answer_text: str) -> List[Dict[str, str]]:
    citations = []
    seen = set()

    for c in chunks:
        source_file = c.get("source_filename", "unknown")
        heading_path = c.get("heading_path", "Root")
        chunk_id = c.get("chunk_id", "")
        key = (source_file, heading_path)

        if key not in seen:
            seen.add(key)
            citations.append({
                "source_file": source_file,
                "heading_path": heading_path,
                "chunk_id": chunk_id
            })

    return citations

def is_greeting(query: str) -> bool:
    q_clean = query.lower().strip().strip("!?.,")
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "help", "who are you", "what can you do", "what is summify", "what is lucidoc"}
    return q_clean in greetings or (len(q_clean.split()) <= 2 and any(g in q_clean for g in ["hi", "hello", "hey", "help"]))

def answer_question(
    session_id: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_client: Optional[LLMClient] = None,
    k: int = 5
) -> DraftAnswer:
    """
    Retrieves top-k relevant chunks from session vector index and generates a grounded answer with citations.
    """
    if llm_client is None:
        llm_client = LLMClient()

    # 1. Handle Greetings & Conversational Queries
    if is_greeting(question):
        return DraftAnswer(
            answer_text="Hello! I am Summify, your AI Documentation Engine. Upload your documents in Tab 1 ('Document Ingestion') to generate structured docs and ask grounded questions!",
            citations=[],
            retrieved_context=[],
            question=question
        )

    # 2. Retrieve Context Chunks
    retrieved_chunks = query_index(session_id, question, k=k)

    # If no chunks exist in session index at all (no docs uploaded yet)
    from pipeline.index import InMemoryVectorStore
    all_chunks = InMemoryVectorStore.get_collection(session_id)
    if not all_chunks and not retrieved_chunks:
        return DraftAnswer(
            answer_text="No documents have been uploaded or indexed for this session yet. Please upload your documentation files in Tab 1 ('Document Ingestion') first!",
            citations=[],
            retrieved_context=[],
            question=question
        )

    # 3. Format Prompt Inputs
    context_text = format_context_for_prompt(retrieved_chunks)
    history_text = format_history_for_prompt(chat_history)

    prompt = CHAT_USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        history_text=history_text,
        question=question
    )

    # 4. Generate Completion
    try:
        raw_answer = llm_client.generate(
            prompt,
            system_prompt=CHAT_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.0
        )
    except Exception as e:
        logger.error(f"Failed to generate RAG answer for session {session_id}: {e}")
        raw_answer = f"I encountered an error generating an answer: {e}"

    # 5. Extract Structured Citations
    citations = extract_citations_from_context(retrieved_chunks, raw_answer)

    return DraftAnswer(
        answer_text=raw_answer,
        citations=citations,
        retrieved_context=retrieved_chunks,
        question=question
    )
