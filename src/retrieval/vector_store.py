from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.ingestion.document_loader import DocumentChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 3
CHROMA_COLLECTION_NAME = "qtrade_support_docs"
DEFAULT_PERSIST_DIR = "./chroma_db"

# BGE models perform better with this query prefix (from the BGE paper)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    """A DocumentChunk paired with its similarity score."""

    chunk: DocumentChunk
    score: float    # similarity score ∈ [-1, 1], where 1 is a better match


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class QTradeVectorStore:
    """
        ChromaDB-backed persistent vector store for QTrade support docs.

        Usage
        -----
        store = QTradeVectorStore()          # uses ./chroma_db by default
        store.index(chunks)                 # embeds and persists chunks
        results = store.retrieve("how do I reset my hub?", top_k=5)

        The store is idempotent, that is calling index() multiple times with the same chunks will not create duplicates.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._collection_name = collection_name

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        # initialize the persistent ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False), # disable telemetry for privacy
        )
        # initialize the persistent ChromaDB collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # use cosine distance, instead of L2
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d documents currently indexed.",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: list[DocumentChunk]) -> None:
        """
            Embed chunks and upsert into the ChromaDB collection.
        """
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        logger.info("Embedding %d chunks with %s …", len(chunks), EMBEDDING_MODEL_NAME)

        texts = [c.text for c in chunks]
        # Encode the texts into embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True, # ensures that the embeddings are unit vectors, which is important for cosine similarity
        ).tolist()

        # Upsert the embeddings and metadata into the ChromaDB collection
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_doc": c.source_doc,
                    "source_file": c.source_file,
                    "char_start": c.char_start,
                }
                for c in chunks
            ],
        )
        logger.info(
            "Upserted %d chunks. Collection now has %d total.",
            len(chunks),
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievedChunk]:
        """
          Return the top_k most relevant chunks for query.

          Returns an empty list when the collection has no documents yet.
        """
        if self._collection.count() == 0:
            logger.warning("Collection is empty, please run ingestion first.")
            return []

        prefixed_query = BGE_QUERY_PREFIX + query
        query_embedding = self._model.encode(
            prefixed_query,
            normalize_embeddings=True,
        ).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine collection returns cosine distance ∈ [0, 2]
            # Convert to similarity ∈ [-1, 1]: similarity = 1 - distance
            score = 1.0 - distance

            retrieved.append(
                RetrievedChunk(
                    chunk=DocumentChunk(
                        chunk_id=results["ids"][0][len(retrieved)],
                        source_doc=meta["source_doc"],
                        source_file=meta["source_file"],
                        text=text,
                        char_start=meta.get("char_start", 0),
                    ),
                    score=score,
                )
            )

        logger.debug(
            "Query: %r — retrieved %d chunks (top score: %.3f)",
            query[:60],
            len(retrieved),
            retrieved[0].score if retrieved else 0.0,
        )
        return retrieved

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """
            Delete and recreate the collection. Useful for testing and
            re-ingestion from scratch. NOTE: it is destructive, so use with care.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' reset.", self._collection_name)

    def __repr__(self) -> str:
        return (
            f"QTradeVectorStore("
            f"model={EMBEDDING_MODEL_NAME!r}, "
            f"collection={self._collection_name!r}, "
            f"chunks={self.num_chunks})"
        )