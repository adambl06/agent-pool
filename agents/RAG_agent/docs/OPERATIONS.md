# RAG Agent Operations

## Environment Variables

### Core LLM

- `LLM_TYPE` (default: `qwen3`)
- `LLM_PROVIDER` (default: `ollama`)
- `LLM_ENDPOINT`
- `LLM_MODEL_NAME`
- `LLM_DISPLAY_NAME`
- `LLM_TIMEOUT_SECONDS` (default: `60`)

### RAG

- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `RAG_TOP_K` (default: `4`)
- `DOCUMENTS_DIR`
- `INGEST_ON_STARTUP` (`true|false`)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RAG_ONLY_LATEST` (`true|false`, default: `true`)
- `SKIP_UNCHANGED_DOCS` (`true|false`, default: `true`)
- `ENRICHMENT_CONFIG_FILE` (optional path, default: `${DOCUMENTS_DIR}/.enrichment.yaml`)

### Routing and Tooling

- `ROUTER_FORCE_RAG` (`true|false`)
- `ROUTER_DOC_KEYWORDS` (comma-separated list)
- `ENABLE_SCHEMA_TOOLS` (`true|false`)

## Startup Behavior

On startup, the service can:

1. Optionally ingest documents (`INGEST_ON_STARTUP=true`).
2. Warm the RAG chain singleton.
3. Warm the general LLM singleton.

## Enrichment and Versioning

The ingestion pipeline supports document enrichment and explicit versions.

1. Per-document metadata added during ingestion:
- `doc_id`
- `doc_version`
- `doc_hash`
- `source_file`
- `source_path`
- `is_latest`

2. Chunk-correlation metadata added after splitting:
- `chunk_id`
- `chunk_index`
- `chunk_total`
- `prev_chunk_id`
- `next_chunk_id`

3. Optional enrichment config file format:

```yaml
defaults:
  prepend: "Global context injected at beginning"
  append: "Global footer appended to all docs"

documents:
  cnip-runbook.pdf:
    version: "v3"
    prepend: "Runbook domain: CNIP deprovisioning"
    append: "Use production-safe procedures only"
```

4. Version handling behavior:
- If `version` is set in enrichment config, that value is used.
- Otherwise a deterministic version is derived from file hash.
- New versions are marked `is_latest=true`.
- Previous versions for same `doc_id` are downgraded to `is_latest=false`.
- Retrieval can be constrained to latest versions via `RAG_ONLY_LATEST=true`.

5. Ingestion optimization:
- With `SKIP_UNCHANGED_DOCS=true`, unchanged document versions are skipped.
- Deterministic chunk IDs avoid duplicate point IDs across re-ingestion.

## Expected Logs

For each request:

1. `Chat request received ...`
2. `Chat routing ... intent=doc|chat`
3. If doc: Qdrant query + LLM call logs
4. `Chat request completed ... sources=<n>`

## Common Issues

### Too many RAG calls for UI helper prompts

- Ensure router excludes metadata prompts (already handled for `### Task:`).

### Slow response latency

- Increase `LLM_TIMEOUT_SECONDS`.
- Tune `RAG_TOP_K` down if retrieval context is too large.

### No document grounding

- Confirm `intent=doc` for relevant queries.
- Verify Qdrant collection and document ingestion logs.
- Verify `RAG_ONLY_LATEST` does not hide intended legacy versions.

## Deployment Notes

- Rebuild image after code changes.
- Redeploy Kubernetes manifests.
- Verify with:
  - one doc-grounded query
  - one generic chat query
  - one OpenWebUI metadata query

Expected outcome:

- doc query => `intent=doc`, `sources>0`
- generic query => `intent=chat`, `sources=0`
- metadata query => `intent=chat`, no retrieval call
