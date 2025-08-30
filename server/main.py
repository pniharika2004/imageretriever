import os
import re
import json
import time
import secrets
import hashlib
from typing import Dict, List, Any, Set, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .db import (
    init_db,
    insert_document,
    insert_pages_bulk,
    get_documents as db_get_documents,
    get_document as db_get_document,
    search_pages as db_search_pages,
    is_fts_enabled,
    get_pages_for_doc,
    get_page_text,
)
from .vector_store import (
    init_vector_store,
    index_doc_pages,
    vector_search,
    is_vector_store_ready,
    has_doc as vec_has_doc,
)

try:
    # Load .env early
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:  # pragma: no cover
    pass

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # Will fallback if not available/running

try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

# Optional reranker
try:  # type: ignore
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

try:
    from pypdf import PdfReader  # modern PyPDF2
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install pypdf (pip install pypdf)") from e

# Optional OCR/Image support (graceful fallback if unavailable)
try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

# Allow configuring external binary paths
POPPLER_PATH = os.environ.get("POPPLER_PATH")
try:
    if pytesseract is not None:
        _tc = os.environ.get("TESSERACT_CMD")
        if _tc:
            # e.g., /opt/homebrew/bin/tesseract
            pytesseract.pytesseract.tesseract_cmd = _tc  # type: ignore
except Exception:
    pass

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
PDFS_DIR = os.path.join(PUBLIC_DIR, "pdfs")
os.makedirs(PDFS_DIR, exist_ok=True)
PREVIEWS_DIR = os.path.join(PUBLIC_DIR, "previews")
os.makedirs(PREVIEWS_DIR, exist_ok=True)


# (Removed) In-memory doc cache; use DB + vector store only
# Initialize SQLite database on startup
@app.on_event("startup")
def _startup() -> None:
    init_db()
    init_vector_store(BASE_DIR)
    # Index any existing PDFs in the public folder if not present in DB
    try:
        for name in os.listdir(PDFS_DIR):
            if not name.lower().endswith(".pdf"):
                continue
            doc_id = os.path.splitext(name)[0]
            # Skip if already present
            if db_get_document(doc_id):
                continue
            pdf_path = os.path.join(PDFS_DIR, name)
            extracted = extract_per_page_text(pdf_path)
            pdf_url = f"/pdfs/{name}"
            insert_document(doc_id, name, pdf_url, extracted["num_pages"])
            insert_pages_bulk(doc_id, extracted["pages"])
            # Index into vector store
            try:
                index_doc_pages(doc_id, extracted["pages"])
            except Exception:
                pass
        # Ensure vector index for any docs that are in DB but missing in vectors
        try:
            for r in db_get_documents():
                did = r.get("doc_id")
                if did and not vec_has_doc(did):
                    pages = get_pages_for_doc(did)
                    if pages:
                        index_doc_pages(did, pages)
        except Exception:
            pass
    except Exception:
        # Best-effort indexing; continue startup
        pass



# OCR configuration via environment (default: always run OCR and merge)
ENABLE_OCR = os.environ.get("ENABLE_OCR", "1") == "1"
OCR_ON_EMPTY_ONLY = os.environ.get("OCR_ON_EMPTY_ONLY", "0") == "1"
try:
    OCR_MIN_TEXT_CHARS = int(os.environ.get("OCR_MIN_TEXT_CHARS", "40"))
except Exception:
    OCR_MIN_TEXT_CHARS = 40
try:
    OCR_DPI = int(os.environ.get("OCR_DPI", "220"))
except Exception:
    OCR_DPI = 220
OCR_LANG = os.environ.get("TESSERACT_LANG", "eng")

# Vision-QA configuration (default: enabled to read images with text)
ENABLE_VISION_QA = os.environ.get("ENABLE_VISION_QA", "1") == "1"
try:
    MAX_VISION_PAGES = int(os.environ.get("MAX_VISION_PAGES", "2"))
except Exception:
    MAX_VISION_PAGES = 2
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")

# Retrieval tuning
try:
    TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "12"))
except Exception:
    TOP_K_RETRIEVE = 12
try:
    TOP_K_CONTEXT = int(os.environ.get("TOP_K_CONTEXT", "8"))
except Exception:
    TOP_K_CONTEXT = 8
ENABLE_HYDE = os.environ.get("ENABLE_HYDE", "1") == "1"
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "1") == "1"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_QUERY_REWRITE = os.environ.get("ENABLE_QUERY_REWRITE", "1") == "1"
try:
    NUM_QUERY_PARAPHRASES = int(os.environ.get("NUM_QUERY_PARAPHRASES", "2"))
except Exception:
    NUM_QUERY_PARAPHRASES = 2

# Image return limits (reduce noise)
try:
    MAX_RETURN_IMAGES = int(os.environ.get("MAX_RETURN_IMAGES", "2"))
except Exception:
    MAX_RETURN_IMAGES = 2
try:
    MAX_IMAGES_PER_DOC = int(os.environ.get("MAX_IMAGES_PER_DOC", "1"))
except Exception:
    MAX_IMAGES_PER_DOC = 1

# Neighbor page expansion toggle (off by default)
EXPAND_NEIGHBORS = os.environ.get("EXPAND_NEIGHBORS", "0") == "1"

# Heuristic control for auto images on visual queries
AUTO_IMAGE_ON_KEYWORDS = os.environ.get("AUTO_IMAGE_ON_KEYWORDS", "1") == "1"
VISUAL_HINT_KEYWORDS = (
    "figure", "fig.", "diagram", "chart", "graph", "table", "image", "screenshot", "plot", "picture"
)


def is_visual_query(query: str) -> bool:
    # Legacy heuristic removed; rely on LLM signals
    return False


def extract_per_page_text(pdf_path: str) -> Dict[str, Any]:
    reader = PdfReader(pdf_path)
    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()

        # OCR fallback/augmentation for image-based pages
        needs_ocr = False
        try:
            needs_ocr = ENABLE_OCR or (OCR_ON_EMPTY_ONLY and len(text) < OCR_MIN_TEXT_CHARS)
        except Exception:
            needs_ocr = False
        if pytesseract is not None and convert_from_path is not None and needs_ocr:
            try:
                _kwargs: Dict[str, Any] = {
                    "first_page": idx,
                    "last_page": idx,
                    "dpi": OCR_DPI,
                    "fmt": "png",
                }
                if POPPLER_PATH:
                    _kwargs["poppler_path"] = POPPLER_PATH
                imgs = convert_from_path(pdf_path, **_kwargs)
                if imgs:
                    ocr_text = pytesseract.image_to_string(imgs[0], lang=OCR_LANG) or ""
                    ocr_text = re.sub(r"\s+", " ", ocr_text).strip()
                    if ocr_text:
                        if text and len(text) >= OCR_MIN_TEXT_CHARS:
                            text = f"{text} {ocr_text}".strip()
                        else:
                            # Prefer OCR text when native text is empty/very short
                            text = ocr_text
            except Exception:
                # Best-effort OCR; ignore errors to keep ingestion robust
                pass
        pages.append({"page_number": idx, "text": text})
    return {"pages": pages, "num_pages": len(pages)}


# --- Retrieval helpers: HyDE expansion and cross-encoder reranking ---
_reranker: Optional["CrossEncoder"] = None


def _get_reranker() -> Optional["CrossEncoder"]:
    global _reranker
    if _reranker is not None:
        return _reranker
    if not ENABLE_RERANK or CrossEncoder is None:
        return None
    try:
        _reranker = CrossEncoder(RERANKER_MODEL)
        return _reranker
    except Exception:
        return None


def hyde_expand_query(question: str) -> Optional[str]:
    if not ENABLE_HYDE:
        return None
    try:
        if Groq is None or not os.environ.get("GROQ_API_KEY"):
            return None
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        prompt = (
            "Write a concise, factual paragraph that would likely appear in the document in response to the question. "
            "Do not reference external knowledge. 120-180 words.\n\nQuestion: " + (question or "")
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You produce a relevant passage, not an answer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content[:1000] if content else None
    except Exception:
        return None


def rewrite_queries(question: str) -> List[str]:
    if not ENABLE_QUERY_REWRITE or not question:
        return []
    try:
        if Groq is None or not os.environ.get("GROQ_API_KEY"):
            return []
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        prompt = (
            f"Rewrite the question into up to {NUM_QUERY_PARAPHRASES} short paraphrases that keep the same meaning. "
            "Do not answer. Return ONLY a JSON object with key 'paraphrases' (array of strings).\n\nQuestion: "
            + (question or "")
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a query rewriting assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        items = data.get("paraphrases") if isinstance(data, dict) else None
        if isinstance(items, list):
            out = []
            for it in items:
                if isinstance(it, str) and it.strip():
                    out.append(it.strip())
                if len(out) >= max(1, NUM_QUERY_PARAPHRASES):
                    break
            return out
        return []
    except Exception:
        return []


def render_pdf_page_to_png(pdf_path: str, page_number: int, dpi: int = 220) -> Optional[str]:
    """Render a single PDF page to a PNG file and return the absolute path.

    Best-effort: returns None if rendering fails or dependencies are missing.
    Caches by filename so repeated calls are cheap.
    """
    try:
        if convert_from_path is None:
            return None
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_name = f"{base_name}-page-{page_number}@{dpi}.png"
        out_path = os.path.join(PREVIEWS_DIR, out_name)
        if os.path.exists(out_path):
            return out_path
        kwargs: Dict[str, Any] = {
            "first_page": page_number,
            "last_page": page_number,
            "dpi": dpi,
            "fmt": "png",
        }
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
        imgs = convert_from_path(pdf_path, **kwargs)
        if not imgs:
            return None
        # pdf2image returns PIL Images; save the first
        img = imgs[0]
        # Ensure previews dir exists
        os.makedirs(PREVIEWS_DIR, exist_ok=True)
        img.save(out_path, format="PNG")
        return out_path
    except Exception:
        return None


def generate_doc_id() -> str:
    return f"{int(time.time()*1000):x}-{secrets.token_hex(4)}"


# API
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    doc_id = generate_doc_id()
    filename = f"{doc_id}.pdf"
    save_path = os.path.join(PDFS_DIR, filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        extracted = extract_per_page_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    pdf_url = f"/pdfs/{filename}"
    # Persist to DB for cross-document search
    try:
        insert_document(doc_id, filename, pdf_url, extracted["num_pages"])
        insert_pages_bulk(doc_id, extracted["pages"])
        # Index into vector store
        try:
            index_doc_pages(doc_id, extracted["pages"])
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save to DB: {e}")
    return {"docId": doc_id, "pdfUrl": pdf_url, "numPages": extracted["num_pages"]}


@app.get("/api/docs")
async def list_docs():
    try:
        rows = db_get_documents()
        items = [
            {"docId": r["doc_id"], "pdfUrl": r["pdf_url"], "numPages": r["num_pages"]}
            for r in rows
        ]
        return {"documents": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@app.post("/api/chat")
async def chat(payload: Dict[str, Any]):
    message = (payload or {}).get("message")
    doc_id = (payload or {}).get("docId")
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    # Gather candidate pages either within a document or across all documents
    candidates: List[Dict[str, Any]] = []  # each: {doc_id, page_number, text}
    doc_meta: Dict[str, Any] | None = None
    used_db = False
    backend: str | None = None
    scope: str = "global" if not doc_id else "single_doc"
    search_backend: str | None = None
    if doc_id:
        # Hybrid retrieve: combine vectors and FTS, then dedupe
        combined: Dict[tuple[str, int], Dict[str, Any]] = {}
        vec_hits: List[Dict[str, Any]] = []
        fts_hits: List[Dict[str, Any]] = []
        # HyDE expansion to improve recall
        hyde = hyde_expand_query(message)
        paraphrases = rewrite_queries(message)
        vector_queries = [message]
        if hyde:
            vector_queries.append(hyde)
        for p in paraphrases:
            if p not in vector_queries:
                vector_queries.append(p)
        if is_vector_store_ready():
            try:
                # Query vectors with original and HyDE text, then merge
                for q in vector_queries:
                    res = vector_search(q, top_k=TOP_K_RETRIEVE, restrict_doc_id=doc_id)
                    for r in res:
                        key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                        if key not in combined:
                            combined[key] = {"doc_id": key[0], "page_number": key[1], "text": r.get("text", "")}
                            vec_hits.append(r)
                for r in vec_hits:
                    key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                    if key not in combined:
                        combined[key] = {"doc_id": key[0], "page_number": key[1], "text": r.get("text", "")}
                backend = "vector_faiss"
            except Exception:
                vec_hits = []
        try:
            fts_hits = db_search_pages(message, limit=TOP_K_RETRIEVE, restrict_doc_id=doc_id)
            for r in fts_hits:
                key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                if key not in combined:
                    combined[key] = {"doc_id": key[0], "page_number": key[1], "text": r.get("text", "")}
            used_db = True
            backend = backend or ("fts5" if is_fts_enabled() else "like")
        except Exception:
            fts_hits = []
        # Prefer vector rank order first, then FTS to fill
        ordered: List[Dict[str, Any]] = []
        seen_keys: Set[tuple[str, int]] = set()
        for r in vec_hits:
            key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
            if key in combined and key not in seen_keys:
                ordered.append(combined[key])
                seen_keys.add(key)
        for r in fts_hits:
            key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
            if key in combined and key not in seen_keys:
                ordered.append(combined[key])
                seen_keys.add(key)
        # Optional cross-encoder rerank
        reranker = _get_reranker()
        if reranker and ordered:
            pairs = [(message, it.get("text", "")) for it in ordered]
            scores = reranker.predict(pairs).tolist()
            ordered = [it for _, it in sorted(zip(scores, ordered), key=lambda x: x[0], reverse=True)]
        candidates = ordered[:TOP_K_CONTEXT]
        search_backend = "hybrid" if vec_hits and fts_hits else (backend or None)
        # Load doc metadata for image URLs
        doc_meta = db_get_document(doc_id)
        if not doc_meta:
            raise HTTPException(status_code=404, detail="Unknown docId")
    else:
        # Global hybrid retrieval
        combined: Dict[tuple[str, int], Dict[str, Any]] = {}
        vec_hits = []
        fts_hits = []
        hyde = hyde_expand_query(message)
        paraphrases = rewrite_queries(message)
        vector_queries = [message]
        if hyde:
            vector_queries.append(hyde)
        for p in paraphrases:
            if p not in vector_queries:
                vector_queries.append(p)
        if is_vector_store_ready():
            try:
                for q in vector_queries:
                    res = vector_search(q, top_k=TOP_K_RETRIEVE)
                    for r in res:
                        key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                        if key not in combined:
                            combined[key] = {"doc_id": key[0], "page_number": key[1], "text": r.get("text", "")}
                            vec_hits.append(r)
                for r in vec_hits:
                    key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                    if key not in combined:
                        combined[key] = {"doc_id": key[0], "page_number": key[1], "text": r.get("text", "")}
                backend = "vector_faiss"
            except Exception:
                vec_hits = []
        try:
            fts_hits = db_search_pages(message, limit=TOP_K_RETRIEVE)
            for r in fts_hits:
                key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                if key not in combined:
                    combined[key] = {"doc_id": key[0], "page_number": key[1], "text": r.get("text", "")}
            used_db = True
            backend = backend or ("fts5" if is_fts_enabled() else "like")
        except Exception:
            fts_hits = []
        ordered = []
        seen_keys: Set[tuple[str, int]] = set()
        for r in vec_hits:
            key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
            if key in combined and key not in seen_keys:
                ordered.append(combined[key])
                seen_keys.add(key)
        for r in fts_hits:
            key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
            if key in combined and key not in seen_keys:
                ordered.append(combined[key])
                seen_keys.add(key)
        reranker = _get_reranker()
        if reranker and ordered:
            pairs = [(message, it.get("text", "")) for it in ordered]
            scores = reranker.predict(pairs).tolist()
            ordered = [it for _, it in sorted(zip(scores, ordered), key=lambda x: x[0], reverse=True)]
        candidates = ordered[:TOP_K_CONTEXT]
        search_backend = "hybrid" if vec_hits and fts_hits else (backend or None)

    # Build context and image mapping
    doc_to_pages: Dict[str, List[int]] = {}
    orig_doc_to_pages: Dict[str, List[int]] = {}
    context_blocks: List[str] = []
    for row in candidates:
        did = str(row.get("doc_id"))
        pn = int(row.get("page_number", 0))
        if pn <= 0:
            continue
        if did not in doc_to_pages:
            doc_to_pages[did] = []
        if did not in orig_doc_to_pages:
            orig_doc_to_pages[did] = []
        if pn not in doc_to_pages[did]:
            doc_to_pages[did].append(pn)
        if pn not in orig_doc_to_pages[did]:
            orig_doc_to_pages[did].append(pn)
        snippet = (row.get("text") or "")[:1200]
        context_blocks.append(f"Doc {did} - Page {pn}:\n{snippet}")
    # Optional: expand with neighboring pages (disabled by default)
    if EXPAND_NEIGHBORS and doc_to_pages:
        for did, pnums in list(doc_to_pages.items()):
            meta = db_get_document(did)
            max_pages = int((meta or {}).get("num_pages") or 0)
            expanded: List[int] = []
            seen_local: Set[int] = set(pnums)
            for pn in pnums:
                for nb in (pn - 1, pn + 1):
                    if nb >= 1 and (max_pages == 0 or nb <= max_pages) and nb not in seen_local:
                        expanded.append(nb)
                        seen_local.add(nb)
            if expanded:
                doc_to_pages[did].extend(expanded)
    if not context_blocks:
        # Soft fallback response when nothing indexed or matched
        return JSONResponse({
            "answer": "I couldn't find relevant content yet. Try uploading a PDF or asking a broader question.",
            "needs_image": False,
            "related_pages": [],
            "pdf_url": None,
            "images": [],
            "search_info": {
                "used_db": used_db,
                "backend": backend,
                "scope": scope,
                "candidate_count": 0,
            },
        })
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided document context. "
        "If an image/figure/diagram/table is directly relevant, set needs_image to true and include related page numbers. "
        f"Return at most {MAX_RETURN_IMAGES} related page numbers (integers) and only those essential for the answer. "
        "Respond in STRICT JSON only with keys: answer (string), needs_image (boolean), related_pages (array of integers). "
        "Do not include any extra commentary."
    )
    user_prompt = f"Question: {message}\n\nDocument context:\n{context}"

    answer = ""
    needs_image = False

    # For backward compatibility when a single doc is targeted
    first_doc_id = next(iter(doc_to_pages.keys())) if doc_to_pages else (doc_id or "")
    related_pages: List[int] = doc_to_pages.get(first_doc_id, []) if first_doc_id else []

    # Prepare optional page images for multimodal models
    images_for_llm_paths: List[str] = []
    if ENABLE_VISION_QA and convert_from_path is not None and MAX_VISION_PAGES > 0:
        pair_list: List[tuple[str, int]] = []
        seen_pairs: Set[str] = set()
        for row in candidates:
            did = str(row.get("doc_id"))
            pn = int(row.get("page_number", 0))
            if pn <= 0:
                continue
            key = f"{did}:{pn}"
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pair_list.append((did, pn))
            if len(pair_list) >= MAX_VISION_PAGES:
                break
        for did, pn in pair_list:
            meta_local = db_get_document(did)
            filename_local = (meta_local or {}).get("filename")
            if not filename_local:
                continue
            pdf_abs_path = os.path.join(PDFS_DIR, filename_local)
            png_path = render_pdf_page_to_png(pdf_abs_path, pn, OCR_DPI)
            if png_path:
                images_for_llm_paths.append(png_path)

    # Try multimodal model first (if configured), then fallback to text-only
    parsed: Optional[Dict[str, Any]] = None
    if ollama is not None and images_for_llm_paths:
        try:
            response = ollama.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt, "images": images_for_llm_paths},
                ],
                options={"temperature": 0.2},
            )
            content = (response.get("message", {}) or {}).get("content", "").strip()
            json_text = content
            m = re.search(r"\{[\s\S]*\}", json_text)
            if m:
                json_text = m.group(0)
            parsed = json.loads(json_text)
        except Exception:
            parsed = None

    if parsed is None:
        # Try Groq text model with JSON output
        try:
            if Groq is None or not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError("groq not available")
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
        except Exception:
            parsed = None

    if parsed is None:
        # Fallback to local Ollama text model
        try:
            if ollama is None:
                raise RuntimeError("ollama not available")
            response = ollama.chat(
                model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.2},
            )
            content = (response.get("message", {}) or {}).get("content", "").strip()
            json_text = content
            m = re.search(r"\{[\s\S]*\}", json_text)
            if m:
                json_text = m.group(0)
            parsed = json.loads(json_text)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        if isinstance(parsed.get("answer"), str):
            answer = parsed["answer"]
        if isinstance(parsed.get("needs_image"), bool):
            needs_image = parsed["needs_image"]
        # related_pages from model only applies within the first doc for compatibility
        if isinstance(parsed.get("related_pages"), list) and parsed["related_pages"] and first_doc_id:
            rp = []
            # Need doc_meta for num_pages when single-doc
            if doc_meta is None and first_doc_id:
                doc_meta = db_get_document(first_doc_id)
            max_pages = (doc_meta or {}).get("num_pages", 0)
            for n in parsed["related_pages"]:
                try:
                    num = int(n)
                    if 1 <= num <= (max_pages or 10_000):
                        rp.append(num)
                except Exception:
                    continue
            if rp:
                # Keep only pages that are in the retrieved candidate set for this doc
                allowed = set(orig_doc_to_pages.get(first_doc_id, []))
                rp_filtered = [p for p in rp if p in allowed] if allowed else rp
                # Cap by MAX_RETURN_IMAGES
                related_pages = rp_filtered[:max(1, min(MAX_RETURN_IMAGES, len(rp_filtered)))] if rp_filtered else []
    if not (isinstance(answer, str) and answer.strip()):
        # Minimal fallback answer when model fails
        # Choose the first candidate snippet
        first_snippet = next((r.get("text", "") for r in candidates if r.get("text")), "")
        answer = first_snippet[:600] if first_snippet else "No clear answer found in the provided pages."
        needs_image = False

    # Ensure we always provide a non-empty answer
    if not (isinstance(answer, str) and answer.strip()):
        first_snippet = next((r.get("text", "") for r in candidates if r.get("text")), "")
        answer = first_snippet[:600] if first_snippet else "No clear answer found in the provided pages."

    # Choose pages for images strictly from LLM output (single-doc)
    should_include_images = bool(first_doc_id and related_pages)

    selected_pages_by_doc: Dict[str, List[int]] = {}
    if should_include_images and first_doc_id and related_pages:
        # Use exactly the model-specified pages (validated earlier)
        selected_pages_by_doc[first_doc_id] = related_pages[:max(1, min(MAX_RETURN_IMAGES, len(related_pages)))]

    # Build images payload with selected pages per document (deduped across near-identical text)
    images: List[Dict[str, Any]] = []
    seen_signatures: Set[str] = set()
    for did, pages in selected_pages_by_doc.items():
        meta = db_get_document(did)
        if not meta:
            continue
        deduped_pages: List[int] = []
        for pn in pages:
            text = get_page_text(did, pn)
            norm = re.sub(r"\s+", " ", (text or ""))[:5000].strip().lower()
            sig = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped_pages.append(pn)
        if deduped_pages:
            images.append({"docId": did, "pdf_url": meta.get("pdf_url"), "pages": deduped_pages})

    # Backward compatibility for existing frontend
    pdf_url_out = None
    if doc_id and doc_meta:
        pdf_url_out = doc_meta.get("pdf_url")
    elif images:
        pdf_url_out = images[0].get("pdf_url")

    # Only mark needs_image when we intentionally selected images
    needs_image = bool(should_include_images and images)

    return JSONResponse({
        "answer": answer,
        "needs_image": needs_image,
        "related_pages": related_pages,
        "pdf_url": pdf_url_out,
        "images": images,
        "search_info": {
            "used_db": used_db,
            "backend": search_backend or backend,
            "scope": scope,
            "candidate_count": len(candidates),
        },
    })


# Static files
# Mount API first, then static at '/'
app.mount("/pdfs", StaticFiles(directory=PDFS_DIR), name="pdfs")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")


