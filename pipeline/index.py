import os
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pipeline.structure import is_noisy_chunk

logger = logging.getLogger(__name__)

# Disable chromadb telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

HAS_CHROMADB = False
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
    HAS_CHROMADB = True
except Exception as e:
    logger.warning(f"chromadb import unavailable ({e}); utilizing in-memory vector index fallback.")

if HAS_CHROMADB:
    class SimpleEmbeddingFunction(EmbeddingFunction[Documents]):
        def _embed_text(self, text: str) -> List[float]:
            text_clean = text.lower().strip()
            words = text_clean.split()
            vec = [0.0] * 128
            
            # Word Unigrams (TF scaling)
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % 128
                vec[idx] += 1.0

            # Character Trigrams for subword & fuzzy match
            if len(text_clean) >= 3:
                for i in range(len(text_clean) - 2):
                    trigram = text_clean[i:i+3]
                    h = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
                    idx = h % 128
                    vec[idx] += 0.5

            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            return [v / norm for v in vec]

        def __call__(self, input: Documents) -> Embeddings:
            return [self._embed_text(t) for t in input]
else:
    class SimpleEmbeddingFunction:
        def _embed_text(self, text: str) -> List[float]:
            text_clean = text.lower().strip()
            words = text_clean.split()
            vec = [0.0] * 128
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % 128
                vec[idx] += 1.0
            if len(text_clean) >= 3:
                for i in range(len(text_clean) - 2):
                    trigram = text_clean[i:i+3]
                    h = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
                    idx = h % 128
                    vec[idx] += 0.5
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            return [v / norm for v in vec]

        def __call__(self, input: List[str]) -> List[List[float]]:
            return [self._embed_text(t) for t in input]

_embed_fn_singleton = None

def get_embedding_function():
    global _embed_fn_singleton
    if _embed_fn_singleton is None:
        _embed_fn_singleton = SimpleEmbeddingFunction()
    return _embed_fn_singleton

# Fallback In-Memory Vector Store
class InMemoryVectorStore:
    _store: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def get_collection(cls, session_id: str) -> List[Dict[str, Any]]:
        return cls._store.setdefault(session_id, [])

    @classmethod
    def clear_collection(cls, session_id: str):
        cls._store[session_id] = []

    @classmethod
    def add(cls, session_id: str, items: List[Dict[str, Any]]):
        coll = cls.get_collection(session_id)
        existing_ids = {it["chunk_id"] for it in coll}
        for item in items:
            if item["chunk_id"] not in existing_ids:
                coll.append(item)
                existing_ids.add(item["chunk_id"])

    @classmethod
    def remove(cls, session_id: str, chunk_ids: Optional[List[str]] = None, source_filename: Optional[str] = None):
        coll = cls.get_collection(session_id)
        if chunk_ids:
            cls._store[session_id] = [it for it in coll if it["chunk_id"] not in chunk_ids]
        elif source_filename:
            cls._store[session_id] = [it for it in coll if it.get("source_filename") != source_filename]

    @classmethod
    def query(cls, session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        coll = cls.get_collection(session_id)
        if not coll:
            return []

        embed_fn = SimpleEmbeddingFunction()
        q_vec = embed_fn([query])[0]
        doc_texts = [it["text"] for it in coll]
        doc_vecs = embed_fn(doc_texts)

        scored = []
        for item, d_vec in zip(coll, doc_vecs):
            dot = sum(a * b for a, b in zip(q_vec, d_vec))
            dist = max(0.0, 1.0 - dot)
            res_item = dict(item)
            res_item["score"] = dist
            scored.append(res_item)

        scored.sort(key=lambda x: x["score"])
        return scored[:k]

def sanitize_collection_name(session_id: str) -> str:
    clean_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    name = f"session_{clean_id}"
    if len(name) < 3:
        name = f"{name}_val"
    return name[:63]

_client_cache: Dict[str, Any] = {}

def _create_chroma_client(storage_dir: Optional[str]):
    if not HAS_CHROMADB:
        return None
    key = storage_dir or "in_memory"
    if key not in _client_cache:
        settings = Settings(anonymized_telemetry=False)
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            _client_cache[key] = chromadb.PersistentClient(path=storage_dir, settings=settings)
        else:
            _client_cache[key] = chromadb.Client(settings=settings)
    return _client_cache[key]

def build_index(
    session_id: str,
    chunks: List[Dict[str, Any]],
    storage_dir: Optional[str] = None
) -> str:
    collection_name = sanitize_collection_name(session_id)

    # Always sync to InMemoryVectorStore as backup vector index
    InMemoryVectorStore.clear_collection(session_id)
    items = []
    for idx, c in enumerate(chunks):
        cid = c.get("chunk_id", f"chunk_{idx}")
        text = c.get("text", "")
        if text.strip() and not is_noisy_chunk(text):
            items.append({
                "session_id": session_id,
                "chunk_id": cid,
                "text": text,
                "source_filename": c.get("source_filename", "unknown"),
                "heading_path": c.get("heading_path", "Root"),
                "chunk_index": c.get("chunk_index", idx),
                "label": c.get("label", "other")
            })
    InMemoryVectorStore.add(session_id, items)

    if HAS_CHROMADB:
        try:
            client = _create_chroma_client(storage_dir)
            embed_fn = get_embedding_function()
            
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.create_collection(name=collection_name, embedding_function=embed_fn)

            ids, docs, metas = [], [], []
            for idx, c in enumerate(chunks):
                cid = c.get("chunk_id", f"chunk_{idx}")
                text = c.get("text", "")
                if not text.strip() or is_noisy_chunk(text):
                    continue
                ids.append(cid)
                docs.append(text)
                metas.append({
                    "session_id": session_id,
                    "chunk_id": cid,
                    "source_filename": c.get("source_filename", "unknown"),
                    "heading_path": c.get("heading_path", "Root"),
                    "chunk_index": c.get("chunk_index", idx),
                    "label": c.get("label", "other")
                })

            if ids:
                collection.add(ids=ids, documents=docs, metadatas=metas)
            return collection_name
        except Exception as e:
            logger.warning(f"ChromaDB build failed ({e}); using InMemoryVectorStore fallback.")

    return collection_name

def add_to_index(
    session_id: str,
    new_chunks: List[Dict[str, Any]],
    storage_dir: Optional[str] = None
) -> None:
    collection_name = sanitize_collection_name(session_id)
    if HAS_CHROMADB:
        try:
            client = _create_chroma_client(storage_dir)
            embed_fn = get_embedding_function()
            collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)
            
            ids, docs, metas = [], [], []
            for idx, c in enumerate(new_chunks):
                cid = c.get("chunk_id", f"new_{idx}")
                text = c.get("text", "")
                if text.strip() and not is_noisy_chunk(text):
                    ids.append(cid)
                    docs.append(text)
                    metas.append({
                        "session_id": session_id,
                        "chunk_id": cid,
                        "source_filename": c.get("source_filename", "unknown"),
                        "heading_path": c.get("heading_path", "Root"),
                        "chunk_index": c.get("chunk_index", idx),
                        "label": c.get("label", "other")
                    })
            if ids:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
            return
        except Exception:
            pass

    items = []
    for idx, c in enumerate(new_chunks):
        cid = c.get("chunk_id", f"new_{idx}")
        text = c.get("text", "")
        if text.strip() and not is_noisy_chunk(text):
            items.append({
                "session_id": session_id,
                "chunk_id": cid,
                "text": text,
                "source_filename": c.get("source_filename", "unknown"),
                "heading_path": c.get("heading_path", "Root"),
                "chunk_index": c.get("chunk_index", idx),
                "label": c.get("label", "other")
            })
    InMemoryVectorStore.add(session_id, items)

def remove_from_index(
    session_id: str,
    chunk_ids: Optional[List[str]] = None,
    source_filename: Optional[str] = None,
    storage_dir: Optional[str] = None
) -> None:
    collection_name = sanitize_collection_name(session_id)
    if HAS_CHROMADB:
        try:
            client = _create_chroma_client(storage_dir)
            embed_fn = get_embedding_function()
            collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
            if chunk_ids:
                collection.delete(ids=chunk_ids)
            elif source_filename:
                collection.delete(where={"source_filename": source_filename})
            return
        except Exception:
            pass

    InMemoryVectorStore.remove(session_id, chunk_ids=chunk_ids, source_filename=source_filename)

def query_index(
    session_id: str,
    query: str,
    k: int = 5,
    storage_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    collection_name = sanitize_collection_name(session_id)
    if HAS_CHROMADB:
        try:
            client = _create_chroma_client(storage_dir)
            embed_fn = get_embedding_function()
            collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
            results = collection.query(query_texts=[query], n_results=k)
            retrieved = []
            if results and "documents" in results and results["documents"]:
                docs_list = results["documents"][0]
                meta_list = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs_list)
                dist_list = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs_list)
                for doc_text, meta, dist in zip(docs_list, meta_list, dist_list):
                    res_item = dict(meta)
                    res_item["text"] = doc_text
                    res_item["score"] = float(dist)
                    retrieved.append(res_item)
            if retrieved:
                return retrieved
        except Exception as e:
            logger.debug(f"Chroma query failed ({e}); trying InMemoryVectorStore.")

    return InMemoryVectorStore.query(session_id, query, k=k)
