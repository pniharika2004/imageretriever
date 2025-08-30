import os
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # type: ignore
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore

# Optional local embedding models via sentence-transformers
try:  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# Optional Postgres + pgvector for large-scale vector search
try:  # type: ignore
    import psycopg  # type: ignore
    from psycopg_pool import ConnectionPool  # type: ignore
    from pgvector.psycopg import register_vector  # type: ignore
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore
    ConnectionPool = None  # type: ignore
    register_vector = None  # type: ignore


# --- Embedding helpers (prefer local sentence-transformers, fallback to Ollama) ---
_st_model: Optional["SentenceTransformer"] = None


def _load_st_model() -> Optional["SentenceTransformer"]:
    global _st_model
    if _st_model is not None:
        return _st_model
    if SentenceTransformer is None:
        return None
    try:
        model_name = os.environ.get("ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _st_model = SentenceTransformer(model_name)
        return _st_model
    except Exception:
        return None


def embed_texts(texts: List[str]) -> Optional["np.ndarray"]:
    # Prefer sentence-transformers if available
    if np is None:
        return None
    try:
        model = _load_st_model()
        if model is not None:
            vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            arr = np.array(vecs, dtype="float32")
            return arr
    except Exception:
        pass
    # Fallback to Ollama embeddings
    if ollama is None:
        return None
    try:
        model_name = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        vectors: List[List[float]] = []
        for t in texts:
            resp = ollama.embeddings(model=model_name, prompt=t)
            e = (resp or {}).get("embedding")
            if isinstance(e, list):
                vectors.append(e)
        if not vectors:
            return None
        arr = np.array(vectors, dtype="float32")
        # Normalize rows for cosine/IP
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms
    except Exception:
        return None


class PGVectorStore:
    def __init__(self) -> None:
        self.url = (
            os.environ.get("PGVECTOR_URL")
            or os.environ.get("POSTGRES_URL")
            or os.environ.get("DATABASE_URL")
        )
        self.pool: Optional[ConnectionPool] = None
        self.dim: Optional[int] = None
        self.table = os.environ.get("PGVECTOR_TABLE", "pdf_chunks")
        self.meta_table = os.environ.get("PGVECTOR_META_TABLE", "pdf_vector_meta")
        self.ready: bool = False
        if self.url and psycopg is not None and ConnectionPool is not None:
            try:
                self.pool = ConnectionPool(self.url, min_size=1, max_size=4, kwargs={"autocommit": True})
                with self.pool.connection() as conn:
                    if register_vector is not None:
                        register_vector(conn)
                    with conn.cursor() as cur:
                        try:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        except Exception:
                            pass
                        cur.execute(
                            f"""
                            CREATE TABLE IF NOT EXISTS {self.meta_table} (
                                id INTEGER PRIMARY KEY CHECK (id = 1),
                                dim INTEGER
                            )
                            """
                        )
                        cur.execute(f"SELECT dim FROM {self.meta_table} WHERE id = 1")
                        row = cur.fetchone()
                        if row and row[0]:
                            self.dim = int(row[0])
                        self.ready = True
            except Exception:
                self.pool = None
                self.ready = False

    def is_ready(self) -> bool:
        return bool(self.ready and self.pool is not None)

    def _ensure_schema(self, dim: int) -> None:
        if not self.is_ready():
            return
        with self.pool.connection() as conn:  # type: ignore[union-attr]
            if register_vector is not None:
                register_vector(conn)
            with conn.cursor() as cur:
                if self.dim != dim:
                    try:
                        cur.execute(f"DROP TABLE IF EXISTS {self.table}")
                    except Exception:
                        pass
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table} (
                            id BIGSERIAL PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            page_number INTEGER NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            embedding VECTOR({dim})
                        )
                        """
                    )
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_doc_page ON {self.table}(doc_id, page_number)")
                    try:
                        cur.execute(
                            f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ivf_cosine ON {self.table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                        )
                    except Exception:
                        pass
                    cur.execute(f"DELETE FROM {self.meta_table} WHERE id = 1")
                    cur.execute(f"INSERT INTO {self.meta_table}(id, dim) VALUES (1, %s)", (dim,))
                    self.dim = dim

    def has_doc(self, doc_id: str) -> bool:
        if not self.is_ready():
            return False
        with self.pool.connection() as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {self.table} WHERE doc_id = %s LIMIT 1", (doc_id,))
                return cur.fetchone() is not None

    def index_document(self, doc_id: str, pages: List[Dict[str, Any]]) -> int:
        if not self.is_ready():
            return 0
        chunks: List[Tuple[int, int, str]] = []
        for page in pages:
            page_num = int(page.get("page_number", 0))
            text = (page.get("text", "") or "").strip()
            if not text:
                continue
            start = 0
            n = len(text)
            chunk_size = 800
            overlap = 200
            cidx = 0
            while start < n:
                end = min(start + chunk_size, n)
                chunk = text[start:end]
                chunks.append((page_num, cidx, chunk))
                cidx += 1
                if end == n:
                    break
                start = max(end - overlap, start + 1)
        if not chunks:
            return 0
        texts = [c[2] for c in chunks]
        arr = embed_texts(texts)
        if arr is None:
            return 0
        dim = int(arr.shape[1])
        self._ensure_schema(dim)
        inserted = 0
        batch_size = 128
        with self.pool.connection() as conn:  # type: ignore[union-attr]
            if register_vector is not None:
                register_vector(conn)
            with conn.cursor() as cur:
                for i in range(0, len(chunks), batch_size):
                    part = chunks[i:i + batch_size]
                    vecs = arr[i:i + batch_size]
                    params = []
                    for (page_num, cidx, content), vec in zip(part, vecs):
                        params.append((doc_id, page_num, cidx, content[:1200], vec.tolist()))
                    cur.executemany(
                        f"INSERT INTO {self.table}(doc_id, page_number, chunk_index, content, embedding) VALUES (%s, %s, %s, %s, %s)",
                        params,
                    )
                    inserted += len(part)
        return inserted

    def search(self, query: str, top_k: int = 6, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.is_ready() or not query.strip():
            return []
        q = embed_texts([query])
        if q is None:
            return []
        vec = q[0].tolist()
        sql = f"SELECT doc_id, page_number, content FROM {self.table}"
        params: List[Any] = []
        where = []
        if restrict_doc_id:
            where.append("doc_id = %s")
            params.append(restrict_doc_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.append(vec)
        params.append(int(top_k) * 1)
        with self.pool.connection() as conn:  # type: ignore[union-attr]
            if register_vector is not None:
                register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                out: List[Dict[str, Any]] = []
                seen: set[Tuple[str, int]] = set()
                for r in rows:
                    did = str(r[0])
                    pn = int(r[1])
                    key = (did, pn)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"doc_id": did, "page_number": pn, "text": r[2]})
                    if len(out) >= top_k:
                        break
                return out


class VectorStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.store_dir = os.path.join(base_dir, "server", "vectorstore")
        os.makedirs(self.store_dir, exist_ok=True)
        self.index_path = os.path.join(self.store_dir, "index.faiss")
        self.meta_path = os.path.join(self.store_dir, "meta.jsonl")
        self.index: Any = None
        self.dim: Optional[int] = None
        self.next_id: int = 1
        self.meta_by_id: Dict[int, Dict[str, Any]] = {}

    def _save_meta(self) -> None:
        tmp = os.path.join(self.store_dir, "meta.tmp.jsonl")
        with open(tmp, "w", encoding="utf-8") as f:
            for _id, meta in self.meta_by_id.items():
                rec = {"id": _id, **meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, self.meta_path)

    def _load_meta(self) -> None:
        self.meta_by_id.clear()
        if not os.path.exists(self.meta_path):
            return
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = int(rec.pop("id"))
                self.meta_by_id[rid] = rec
        if self.meta_by_id:
            self.next_id = max(self.meta_by_id.keys()) + 1

    def load_or_init(self) -> None:
        if faiss is None or np is None:
            return
        self._load_meta()
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            try:
                # If not ID mapped, wrap it
                if not isinstance(self.index, faiss.IndexIDMap2):
                    idmap = faiss.IndexIDMap2(self.index)
                    self.index = idmap
            except Exception:
                pass
            # Try to infer dim
            try:
                self.dim = self.index.d
            except Exception:
                self.dim = None
        else:
            self.index = None
            self.dim = None

    def save(self) -> None:
        if faiss is None or self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        self._save_meta()

    def _ensure_index(self, dim: int) -> None:
        if self.index is not None:
            return
        # Cosine similarity via inner product on normalized vectors
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)
        self.dim = dim

    def _normalize_rows(self, x: "np.ndarray") -> "np.ndarray":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def _embed(self, texts: List[str]) -> Optional["np.ndarray"]:
        # Route to global helper for consistent embeddings across backends
        return embed_texts(texts)

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def index_document(self, doc_id: str, pages: List[Dict[str, Any]]) -> int:
        if faiss is None or np is None:
            return 0
        added = 0
        for page in pages:
            page_num = int(page.get("page_number", 0))
            text = page.get("text", "")
            chunks = self._chunk_text(text)
            if not chunks:
                continue
            emb = self._embed(chunks)
            if emb is None:
                return added
            self._ensure_index(emb.shape[1])
            ids = []
            for _ in range(emb.shape[0]):
                ids.append(self.next_id)
                self.next_id += 1
            id_arr = np.array(ids, dtype="int64")
            self.index.add_with_ids(emb, id_arr)
            for cid, chunk_text in zip(ids, chunks):
                self.meta_by_id[cid] = {
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "text": chunk_text[:1200],
                }
            added += len(ids)
        if added:
            self.save()
        return added

    def search(self, query: str, top_k: int = 6, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if faiss is None or np is None or self.index is None or not query.strip():
            return []
        q = self._embed([query])
        if q is None:
            return []
        D, I = self.index.search(q, top_k * 4)
        ids = I[0]
        results: List[Dict[str, Any]] = []
        seen: set[Tuple[str, int]] = set()
        for cid in ids:
            if int(cid) <= 0:
                continue
            meta = self.meta_by_id.get(int(cid))
            if not meta:
                continue
            if restrict_doc_id and meta.get("doc_id") != restrict_doc_id:
                continue
            key = (str(meta.get("doc_id")), int(meta.get("page_number", 0)))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "doc_id": meta.get("doc_id"),
                "page_number": int(meta.get("page_number", 0)),
                "text": meta.get("text", ""),
            })
            if len(results) >= top_k:
                break
        return results

    def has_doc(self, doc_id: str) -> bool:
        for meta in self.meta_by_id.values():
            if meta.get("doc_id") == doc_id:
                return True
        return False


_store: Optional[VectorStore] = None
_pg_store: Optional["PGVectorStore"] = None
_router: Optional["HybridVectorRouter"] = None


def init_vector_store(base_dir: str) -> None:
    global _store, _pg_store, _router
    vs = VectorStore(base_dir)
    vs.load_or_init()
    _store = vs
    pg = PGVectorStore()
    _pg_store = pg if pg.is_ready() else None
    _router = HybridVectorRouter(_store, _pg_store)


def index_doc_pages(doc_id: str, pages: List[Dict[str, Any]]) -> int:
    if _router is None:
        return 0
    return _router.index_document(doc_id, pages)


def vector_search(query: str, top_k: int = 6, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if _router is None:
        return []
    return _router.search(query, top_k=top_k, restrict_doc_id=restrict_doc_id)


def is_vector_store_ready() -> bool:
    if _router is None:
        return False
    return _router.is_ready()


def has_doc(doc_id: str) -> bool:
    if _router is None:
        return False
    return _router.has_doc(doc_id)


class HybridVectorRouter:
    def __init__(self, faiss_store: Optional[VectorStore], pg_store: Optional[PGVectorStore]) -> None:
        self.faiss_store = faiss_store
        self.pg_store = pg_store

    def is_ready(self) -> bool:
        if self.pg_store and self.pg_store.is_ready():
            return True
        return bool(self.faiss_store is not None and self.faiss_store.index is not None)

    def index_document(self, doc_id: str, pages: List[Dict[str, Any]]) -> int:
        if self.pg_store and self.pg_store.is_ready():
            return self.pg_store.index_document(doc_id, pages)
        if self.faiss_store is not None:
            return self.faiss_store.index_document(doc_id, pages)
        return 0

    def search(self, query: str, top_k: int = 6, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.pg_store and self.pg_store.is_ready():
            hits = self.pg_store.search(query, top_k=top_k, restrict_doc_id=restrict_doc_id)
            if hits:
                return hits
        if self.faiss_store is not None:
            return self.faiss_store.search(query, top_k=top_k, restrict_doc_id=restrict_doc_id)
        return []

    def has_doc(self, doc_id: str) -> bool:
        if self.pg_store and self.pg_store.is_ready():
            return self.pg_store.has_doc(doc_id)
        if self.faiss_store is not None:
            return self.faiss_store.has_doc(doc_id)
        return False


