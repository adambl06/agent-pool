## RAG Agent Optimizations

The RAG Agent includes several advanced optimizations for efficient document management and retrieval:

### Document Versioning & Change Detection
- **SHA256-based hashing**: Each document is hashed to detect changes. Only modified documents are re-indexed, significantly reducing unnecessary processing.
- **Deterministic document IDs**: Generated from file paths to prevent collisions and ensure stable references.
- **Version tracking**: Documents can have explicit versions (configured via enrichment config) or automatic versions derived from file hash.
- **Latest-version flag**: Only the newest version of a document is marked as `is_latest=true` for retrieval, previous versions are automatically downgraded.

### Metadata Enrichment System
- **YAML configuration**: Optional `.enrichment.yaml` file in the documents directory allows per-document and global enrichment settings.
- **Flexible content enhancement**: Prepend/append capabilities to add context, headers, or footers to document chunks.
- **Per-document customization**: Each document can have custom version strings and enrichment metadata that are preserved through the ingestion pipeline.
- **Comprehensive metadata tracking**: Each chunk includes:
  - `doc_id`, `doc_version`, `doc_hash` for version management
  - `source_file`, `source_path`, `source_ext` for traceability
  - `is_latest` flag for filtering to current versions
  - `enriched` flag to indicate custom content additions

### Intelligent Chunking & Context Preservation
- **Adjacency-aware linking**: Chunks within the same document are linked via `prev_chunk_id` and `next_chunk_id` for adjacency-aware retrieval.
- **Chunk positioning**: Each chunk tracks its `chunk_index` and `chunk_total` within the document, enabling context-aware queries.
- **Overlapping chunks**: Configurable chunk size and overlap via `CHUNK_SIZE` and `CHUNK_OVERLAP` environment variables.

### Smart Upsert Strategy
- **Skip unchanged documents**: When `SKIP_UNCHANGED_DOCS=true` (default), documents with identical hashes are not re-embedded, reducing embedding costs.
- **Intelligent deduplication**: Documents are grouped by doc_id, version, and hash; duplicates are detected before upsert.
- **Automatic version downgrade**: When a new version of a document is ingested, previous versions are automatically marked as `is_latest=false` in the vector database.
- **Efficient filtering**: Qdrant queries use metadata filters to skip unchanged documents without full collection scans.
