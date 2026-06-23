import hashlib
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models import LLM
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-vector-db.agentic.svc.cluster.local:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")
EMBEDDING_DIM = 768  # all-mpnet-base-v2 output dimension


def _compute_file_hash(file_path: str) -> str:
    """Compute stable SHA256 hash for document versioning and change detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _derive_doc_id(file_path: str) -> str:
    """Create a deterministic document ID from relative path to avoid collisions."""
    filename = Path(file_path).name
    return f"doc::{hashlib.sha1(file_path.encode('utf-8')).hexdigest()[:12]}::{filename}"


def _load_enrichment_config(doc_dir: str) -> Dict[str, Any]:
    """
    Load optional enrichment config from ENRICHMENT_CONFIG_FILE.

    Default path: <DOCUMENTS_DIR>/.enrichment.yaml
    Example schema:
      defaults:
        prepend: "global intro"
        append: "global footer"
      documents:
        runbook.pdf:
          version: "v2"
          prepend: "document-specific header"
          append: "document-specific footer"
    """
    enrichment_file = os.getenv("ENRICHMENT_CONFIG_FILE", os.path.join(doc_dir, ".enrichment.yaml"))
    if not os.path.exists(enrichment_file):
        return {"defaults": {}, "documents": {}}
    try:
        with open(enrichment_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("defaults", {})
        data.setdefault("documents", {})
        logger.info("Loaded enrichment config file=%s", enrichment_file)
        return data
    except Exception:
        logger.exception("Failed to parse enrichment config file=%s", enrichment_file)
        return {"defaults": {}, "documents": {}}


def _get_doc_enrichment(enrichment_cfg: Dict[str, Any], filename: str) -> Dict[str, Any]:
    defaults = enrichment_cfg.get("defaults", {}) or {}
    per_doc = (enrichment_cfg.get("documents", {}) or {}).get(filename, {}) or {}
    return {
        "prepend": f"{defaults.get('prepend', '')}\n{per_doc.get('prepend', '')}".strip(),
        "append": f"{per_doc.get('append', '')}\n{defaults.get('append', '')}".strip(),
        "version": per_doc.get("version"),
    }


# --- Load Documents ---
def load_documents() -> list:
    """
    Here this function is made to load supported source files from DOCUMENTS_DIR 
    into LangChain documents.

    Currently supports txt and pdf files organized in category folders.
    Each folder name becomes a category/knowledge_domain metadata field.
    Logs per-file load status so ingestion issues are easy to diagnose from pod logs.
    """
    doc_dir = os.getenv("DOCUMENTS_DIR", "./documents")
    logger.info(f"Loading documents from {doc_dir}")
    enrichment_cfg = _load_enrichment_config(doc_dir)

    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
    }

    docs = []
    scanned = 0
    
    # Check if doc_dir exists
    if not os.path.exists(doc_dir):
        logger.warning(f"Documents directory does not exist: {doc_dir}")
        return docs
    
    # Iterate over folders in DOCUMENTS_DIR
    for folder_name in os.listdir(doc_dir):
        folder_path = os.path.join(doc_dir, folder_name)
        
        # Skip if not a directory or if it's a hidden folder
        if not os.path.isdir(folder_path) or folder_name.startswith("."):
            continue
        
        logger.info(f"Scanning folder: {folder_name}")
        
        # Iterate over files in each folder
        for filename in os.listdir(folder_path):
            scanned += 1
            file_path = os.path.join(folder_path, filename)
            
            # Skip if it's a directory
            if os.path.isdir(file_path):
                continue
            
            ext = os.path.splitext(filename)[1].lower()
            if ext in loaders:
                try:
                    file_hash = _compute_file_hash(file_path)
                    doc_id = _derive_doc_id(file_path)
                    enrich = _get_doc_enrichment(enrichment_cfg, filename)
                    doc_version = enrich.get("version") or file_hash[:12]

                    loader = loaders[ext](file_path)
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        base_content = d.page_content or ""
                        prepend = enrich.get("prepend", "")
                        append = enrich.get("append", "")
                        if prepend:
                            base_content = f"{prepend}\n\n{base_content}"
                        if append:
                            base_content = f"{base_content}\n\n{append}"

                        d.page_content = base_content
                        d.metadata = d.metadata or {}
                        d.metadata.update(
                            {
                                "doc_id": doc_id,
                                "doc_version": doc_version,
                                "doc_hash": file_hash,
                                "source_file": filename,
                                "source_path": file_path,
                                "source_ext": ext,
                                "knowledge_domain": folder_name,
                                "is_latest": True,
                                "enriched": bool(prepend or append),
                            }
                        )
                    docs.extend(loaded_docs)
                    logger.info(
                        "Loaded document source=%s category=%s doc_id=%s version=%s hash=%s pages=%d enriched=%s",
                        filename,
                        folder_name,
                        doc_id,
                        doc_version,
                        file_hash[:12],
                        len(loaded_docs),
                        bool(enrich.get("prepend") or enrich.get("append")),
                    )
                except Exception as e:
                    logger.error(f"Failed to load {filename} from folder {folder_name}: {e}")

    if not docs:
        logger.warning("No documents loaded!")
    logger.info("Document scan complete files_scanned=%d docs_loaded=%d", scanned, len(docs))
    return docs


# --- Split Documents into Chunks ---
def split_documents(docs: list) -> list:
    """
    Here we Split loaded documents into overlapping text chunks for embedding.
    We want to shorten the text so the embeddings are more focused and relevant to the query.
    The overlap helps maintain context across chunks. To enhence coherence we will also add metadata.
    Chunk size and overlap are controlled through environment variables to
    tune retrieval quality vs latency.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
    )
    chunks = text_splitter.split_documents(docs)

    # Correlate chunks within the same document version for adjacency-aware retrieval.
    by_doc_version: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        md = chunk.metadata or {}
        key = (str(md.get("doc_id", "unknown")), str(md.get("doc_version", "unknown")))
        by_doc_version[key].append(idx)

    for (doc_id, doc_version), indices in by_doc_version.items():
        total = len(indices)
        for pos, chunk_idx in enumerate(indices):
            chunk = chunks[chunk_idx]
            chunk_id = f"{doc_id}:{doc_version}:{pos}"
            prev_chunk_id = f"{doc_id}:{doc_version}:{pos - 1}" if pos > 0 else None
            next_chunk_id = f"{doc_id}:{doc_version}:{pos + 1}" if pos < total - 1 else None
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": pos,
                    "chunk_total": total,
                    "prev_chunk_id": prev_chunk_id,
                    "next_chunk_id": next_chunk_id,
                }
            )

    for (doc_id, doc_version), indices in by_doc_version.items():
        logger.info(
            "Chunk map doc_id=%s version=%s chunks=%d first_chunk=%s last_chunk=%s",
            doc_id,
            doc_version,
            len(indices),
            chunks[indices[0]].metadata.get("chunk_id"),
            chunks[indices[-1]].metadata.get("chunk_id"),
        )

    logger.info("Split complete input_docs=%d output_chunks=%d", len(docs), len(chunks))
    return chunks


# --- Create Vector Store ---
def create_vector_store(chunks: list) -> QdrantVectorStore:
    """
    Embed chunked documents and upsert vectors into Qdrant.

    The collection is created automatically if it does not exist.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Ensure the collection exists
    logger.info(
        "Vector upsert start qdrant_url=%s collection=%s chunks=%d",
        QDRANT_URL,
        QDRANT_COLLECTION,
        len(chunks),
    )
    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'")

    skip_unchanged = os.getenv("SKIP_UNCHANGED_DOCS", "true").lower() == "true"

    # Group chunks by document version for dedup and latest-version flag maintenance.
    grouped: Dict[Tuple[str, str, str], List[Any]] = defaultdict(list)
    for c in chunks:
        md = c.metadata or {}
        grouped[(str(md.get("doc_id")), str(md.get("doc_version")), str(md.get("doc_hash")))].append(c)

    chunks_to_upsert: List[Any] = []
    skipped_docs = 0
    skipped_chunks = 0
    for (doc_id, doc_version, doc_hash), doc_chunks in grouped.items():
        logger.info(
            "Ingest candidate doc_id=%s version=%s hash=%s chunks=%d",
            doc_id,
            doc_version,
            doc_hash[:12],
            len(doc_chunks),
        )
        if skip_unchanged:
            existing_points, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="doc_version", match=MatchValue(value=doc_version)),
                        FieldCondition(key="doc_hash", match=MatchValue(value=doc_hash)),
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if existing_points:
                logger.info(
                    "Skipping unchanged document doc_id=%s version=%s hash=%s chunks=%d",
                    doc_id,
                    doc_version,
                    doc_hash[:12],
                    len(doc_chunks),
                )
                skipped_docs += 1
                skipped_chunks += len(doc_chunks)
                continue

        chunks_to_upsert.extend(doc_chunks)

    logger.info(
        "Ingest planning total_docs=%d skipped_docs=%d skipped_chunks=%d upsert_chunks=%d",
        len(grouped),
        skipped_docs,
        skipped_chunks,
        len(chunks_to_upsert),
    )

    if not chunks_to_upsert:
        logger.info("No new chunks to upsert after unchanged-document filtering")
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )

    chunk_ids = [c.metadata.get("chunk_id") for c in chunks_to_upsert]
    vector_db = QdrantVectorStore.from_documents(
        documents=chunks_to_upsert,
        embedding=embeddings,
        ids=chunk_ids,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )

    # Mark previous versions as not latest for each ingested document.
    for doc_id, doc_version, _ in {(
        str(c.metadata.get("doc_id")),
        str(c.metadata.get("doc_version")),
        str(c.metadata.get("doc_hash")),
    ) for c in chunks_to_upsert}:
        try:
            old_points, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="is_latest", match=MatchValue(value=True)),
                    ],
                    must_not=[
                        FieldCondition(key="doc_version", match=MatchValue(value=doc_version)),
                    ],
                ),
                limit=10_000,
                with_payload=False,
                with_vectors=False,
            )

            old_ids = [p.id for p in old_points if p.id is not None]
            if old_ids:
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload={"is_latest": False},
                    points=old_ids,
                )
                logger.info(
                    "Downgraded previous latest versions doc_id=%s keep_version=%s downgraded_points=%d",
                    doc_id,
                    doc_version,
                    len(old_ids),
                )
        except Exception:
            logger.exception("Failed to downgrade previous latest versions doc_id=%s", doc_id)

    logger.info(
        "Upserted %d chunks into Qdrant collection '%s' sample_chunk_id=%s",
        len(chunks_to_upsert),
        QDRANT_COLLECTION,
        chunks_to_upsert[0].metadata.get("chunk_id") if chunks_to_upsert else None,
    )
    return vector_db


# --- Load Vector Store ---
def load_vector_store() -> QdrantVectorStore:
    """Connect to the existing Qdrant collection for retrieval operations."""
    logger.info("Loading vector store qdrant_url=%s collection=%s", QDRANT_URL, QDRANT_COLLECTION)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )


# --- Build RAG Chain ---
def get_rag_chain(llm: LLM, vector_db: QdrantVectorStore) -> RetrievalQA:
    """
    Build a RetrievalQA chain that does retrieve-then-generate.

    This chain is the core of grounded answering for internal documents.
    """
    k = int(os.getenv("RAG_TOP_K", "4"))
    latest_only = os.getenv("RAG_ONLY_LATEST", "true").lower() == "true"

    search_kwargs: Dict[str, Any] = {"k": k}
    if latest_only:
        search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="is_latest", match=MatchValue(value=True))]
        )

    logger.info("Creating retriever k=%d", k)
    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


# --- Ingest Documents (run once to build the index) ---
def ingest_documents():
    """
    Full indexing pipeline used at startup or manual refresh:
    load documents -> split chunks -> embed -> store in Qdrant.
    """
    docs = load_documents()
    if not docs:
        logger.warning("Ingestion skipped: no documents found.")
        return
    chunks = split_documents(docs)
    create_vector_store(chunks)
    logger.info("Document ingestion complete!")


# --- Query the RAG Pipeline ---
def query_rag(llm: LLM, query: str) -> str:
    """
    Execute a one-shot RAG query and return only the final answer text.

    Source count is logged for observability and retrieval quality checks.
    """
    logger.info("RAG query start query=%s", query[:200])
    vector_db = load_vector_store()
    rag_chain = get_rag_chain(llm, vector_db)
    result = rag_chain.invoke({"query": query})
    sources = result.get("source_documents", [])
    source_ids = [
        (s.metadata or {}).get("chunk_id")
        for s in sources
    ]
    logger.info(
        "RAG query complete source_docs=%d source_chunk_ids=%s answer_chars=%d",
        len(sources),
        source_ids,
        len(result.get("result", "")),
    )
    return result["result"]
