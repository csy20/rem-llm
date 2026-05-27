"""ChromaDB integration for codebase chunk storage and semantic retrieval.

All imports are lazy — nothing is imported until functions are called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def index_to_chromadb(
    project_dir: Path,
    db_path: str = "models/chroma_codebase",
    collection_name: str = "code_chunks",
) -> int:
    import json

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=db_path, settings=Settings(anonymized_telemetry=False)
    )
    collections = [c.name for c in client.list_collections()]
    if collection_name in collections:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    from remllm.context.indexer import CodebaseIndexer

    indexer = CodebaseIndexer(Path("models/codebase_index.json"))
    indexer.index(project_dir)

    if not indexer.chunks:
        return 0

    ids = [f"{c.path}:{c.start_line}:{c.name}" for c in indexer.chunks]
    documents = [c.content for c in indexer.chunks]
    metadatas = [
        {
            "path": c.path,
            "name": c.name,
            "chunk_type": c.chunk_type,
            "start_line": c.start_line,
        }
        for c in indexer.chunks
    ]
    embeddings = [c.embedding for c in indexer.chunks]

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )

    return len(ids)


def search_chromadb(
    query: str,
    db_path: str = "models/chroma_codebase",
    collection_name: str = "code_chunks",
    top_k: int = 5,
) -> list[dict]:
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=db_path, settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    from remllm.context.indexer import CodebaseIndexer

    indexer = CodebaseIndexer()
    query_embed = indexer._embed_text(query)

    results = collection.query(query_embeddings=[query_embed], n_results=top_k)

    chunks = []
    if results and results.get("metadatas") and results["metadatas"][0]:
        for i, meta in enumerate(results["metadatas"][0]):
            doc = results["documents"][0][i] if results.get("documents") else ""
            chunks.append(
                {
                    "path": meta.get("path", ""),
                    "name": meta.get("name", ""),
                    "chunk_type": meta.get("chunk_type", ""),
                    "start_line": meta.get("start_line", 0),
                    "content": doc,
                }
            )
    return chunks
