# rag_tools/rag.py
import os
import logging
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.language_models import LLM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-vector-db.agentic.svc.cluster.local:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")
EMBEDDING_DIM = 768  # all-mpnet-base-v2 output dimension


# --- Load Documents ---
def load_documents() -> list:
    """Load all documents from the configured directory."""
    doc_dir = os.getenv("DOCUMENTS_DIR", "./documents")
    logger.info(f"Loading documents from {doc_dir}")

    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
    }

    docs = []
    scanned = 0
    for filename in os.listdir(doc_dir):
        scanned += 1
        file_path = os.path.join(doc_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext in loaders:
            try:
                loader = loaders[ext](file_path)
                docs.extend(loader.load())
                logger.info(f"Loaded {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

    if not docs:
        logger.warning("No documents loaded!")
    logger.info("Document scan complete files_scanned=%d docs_loaded=%d", scanned, len(docs))
    return docs


# --- Split Documents into Chunks ---
def split_documents(docs: list) -> list:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
    )
    chunks = text_splitter.split_documents(docs)
    logger.info("Split complete input_docs=%d output_chunks=%d", len(docs), len(chunks))
    return chunks


# --- Create Vector Store ---
def create_vector_store(chunks: list) -> QdrantVectorStore:
    """Embed chunks and upsert them into the Qdrant collection."""
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

    vector_db = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )
    logger.info(f"Upserted {len(chunks)} chunks into Qdrant collection '{QDRANT_COLLECTION}'")
    return vector_db


# --- Load Vector Store ---
def load_vector_store() -> QdrantVectorStore:
    """Connect to an existing Qdrant collection."""
    logger.info("Loading vector store qdrant_url=%s collection=%s", QDRANT_URL, QDRANT_COLLECTION)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )


# --- Build RAG Chain ---
def get_rag_chain(llm: LLM, vector_db: QdrantVectorStore) -> RetrievalQA:
    """Create a RetrievalQA chain: retrieves relevant chunks, then feeds them to the LLM."""
    k = int(os.getenv("RAG_TOP_K", "4"))
    logger.info("Creating retriever k=%d", k)
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


# --- Ingest Documents (run once to build the index) ---
def ingest_documents():
    """Load, split, and embed documents into the vector store."""
    docs = load_documents()
    if not docs:
        logger.warning("Ingestion skipped: no documents found.")
        return
    chunks = split_documents(docs)
    create_vector_store(chunks)
    logger.info("Document ingestion complete!")


# --- Query the RAG Pipeline ---
def query_rag(llm: LLM, query: str) -> str:
    """Run a query against the vector store using the given LLM."""
    logger.info("RAG query start query=%s", query[:200])
    vector_db = load_vector_store()
    rag_chain = get_rag_chain(llm, vector_db)
    result = rag_chain({"query": query})
    sources = result.get("source_documents", [])
    logger.info("RAG query complete source_docs=%d answer_chars=%d", len(sources), len(result.get("result", "")))
    return result["result"]
