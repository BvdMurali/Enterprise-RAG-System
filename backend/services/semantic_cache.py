import logging
import sqlite3
import numpy as np
import json
from typing import Optional, Dict, Any
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    Enterprise Semantic Cache using SQLite and local embedding distance comparisons.
    Provides sub-10ms response times for repeating query semantics.
    """

    def __init__(self, embeddings: Embeddings, db_path: str = "./semantic_cache.db", threshold: float = 0.08):
        self.embeddings = embeddings
        self.db_path = db_path
        self.threshold = threshold
        self._init_db()
        logger.info(f"Initialized SemanticCacheService with db_path={db_path}, threshold={threshold}")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    )
                """)
        finally:
            conn.close()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check semantic cache for similar queries.
        Returns the cached answer and sources if query distance is under threshold.
        """
        try:
            query_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed query for cache check: {e}")
            return None

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT query, response, sources_json, embedding FROM semantic_cache")
            rows = cursor.fetchall()
            
            best_match = None
            min_distance = float("inf")

            for cached_query, cached_resp, sources_json, emb_bytes in rows:
                cached_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                # Compute Euclidean distance between normalized embeddings (equivalent to Cosine distance)
                distance = np.linalg.norm(query_emb - cached_emb)
                
                if distance < self.threshold and distance < min_distance:
                    min_distance = distance
                    try:
                        sources = json.loads(sources_json)
                    except Exception:
                        sources = []
                    best_match = {
                        "answer": cached_resp,
                        "sources": sources,
                        "cache_hit": True,
                        "distance": float(round(distance, 4))
                    }

            if best_match:
                logger.info(f"Semantic Cache HIT for query: '{query}' (distance: {min_distance:.4f})")
                return best_match

        except Exception as e:
            logger.error(f"Error reading from semantic cache: {e}", exc_info=True)
        finally:
            conn.close()

        logger.info(f"Semantic Cache MISS for query: '{query}'")
        return None

    def set(self, query: str, response: str, sources: list) -> None:
        """
        Save a query, its response, and its embedding to the semantic cache.
        """
        try:
            query_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
            emb_bytes = query_emb.tobytes()
        except Exception as e:
            logger.error(f"Failed to embed query for cache store: {e}")
            return

        sources_json = json.dumps(sources)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO semantic_cache (query, response, sources_json, embedding) VALUES (?, ?, ?, ?)",
                    (query, response, sources_json, emb_bytes)
                )
            logger.info(f"Cached response for query: '{query}'")
        except Exception as e:
            logger.error(f"Failed to write query to semantic cache: {e}", exc_info=True)
        finally:
            conn.close()

    def clear(self):
        """Delete all cached items."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM semantic_cache")
            logger.warning("Semantic cache cleared successfully.")
        finally:
            conn.close()
