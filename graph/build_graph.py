import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from pipeline.ingest import ingest_files
from pipeline.chunk import chunk_files
from pipeline.classify import classify_chunks
from pipeline.structure import assemble_document
from pipeline.index import build_index
from pipeline.chat import answer_question
from pipeline.faithfulness import verify_and_refine_answer

logger = logging.getLogger(__name__)

class LucidocState(TypedDict, total=False):
    uploaded_files: List[str]
    raw_text: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    classified_chunks: List[Dict[str, Any]]
    structured_document: Dict[str, Any]
    vector_index_ref: Optional[str]
    chat_history: List[Dict[str, Any]]
    session_id: str

def ingest_node(state: LucidocState) -> LucidocState:
    logger.info("Executing Graph Node: ingest")
    uploaded = state.get("uploaded_files", [])
    if uploaded:
        docs = ingest_files(uploaded)
        state["raw_text"] = [doc.to_dict() for doc in docs]
    else:
        state["raw_text"] = []
    return state

def chunk_classify_node(state: LucidocState) -> LucidocState:
    logger.info("Executing Graph Node: chunk_classify")
    raw_docs = state.get("raw_text", [])
    if raw_docs:
        chunks = chunk_files(raw_docs)
        state["chunks"] = [c.to_dict() for c in chunks]
        classified = classify_chunks(state["chunks"])
        state["classified_chunks"] = [cc.to_dict() for cc in classified]
    else:
        state["chunks"] = []
        state["classified_chunks"] = []
    return state

def structure_export_node(state: LucidocState) -> LucidocState:
    logger.info("Executing Graph Node: structure_export")
    classified = state.get("classified_chunks", [])
    if classified:
        doc = assemble_document(classified)
        state["structured_document"] = doc.to_dict()
    else:
        state["structured_document"] = {}
    return state

def index_node(state: LucidocState) -> LucidocState:
    logger.info("Executing Graph Node: index")
    session_id = state.get("session_id", "default_session")
    classified = state.get("classified_chunks", [])
    if classified:
        coll_name = build_index(session_id, classified)
        state["vector_index_ref"] = coll_name
    else:
        state["vector_index_ref"] = ""
    return state

def chat_node(state: LucidocState) -> LucidocState:
    logger.info("Executing Graph Node: chat")
    session_id = state.get("session_id", "default_session")
    history = state.get("chat_history", [])
    
    latest_question = None
    if history and history[-1].get("role") == "user":
        latest_question = history[-1].get("content")

    if latest_question:
        draft = answer_question(session_id, latest_question, chat_history=history[:-1])
        verified_draft = verify_and_refine_answer(session_id, latest_question, draft, chat_history=history[:-1])
        state.setdefault("chat_history", []).append({
            "role": "assistant",
            "content": verified_draft.answer_text,
            "citations": verified_draft.citations
        })
    return state

def build_graph():
    """
    Builds and compiles the LangGraph pipeline for Lucidoc.
    Sequence: ingest -> chunk_classify -> structure_export -> index -> chat
    """
    builder = StateGraph(LucidocState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("chunk_classify", chunk_classify_node)
    builder.add_node("structure_export", structure_export_node)
    builder.add_node("index", index_node)
    builder.add_node("chat", chat_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "chunk_classify")
    builder.add_edge("chunk_classify", "structure_export")
    builder.add_edge("structure_export", "index")
    builder.add_edge("index", "chat")
    builder.add_edge("chat", END)

    return builder.compile()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph = build_graph()
    initial_state: LucidocState = {
        "uploaded_files": ["sample.md"],
        "session_id": "test_session_001"
    }
    result = graph.invoke(initial_state)
    print("Execution Result:", result)
