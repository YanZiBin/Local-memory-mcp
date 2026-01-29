"""Search engine service with hybrid search and RRF fusion."""
import logging
from typing import Dict, List, Optional, Any

import numpy as np

from ranking import calculate_rrf_score, fuse_rankings, distance_to_similarity


DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.7
DEFAULT_RRF_K = 60
FETCH_MULTIPLIER = 4


class SearchService:
    """Encapsulates hybrid search logic with RRF fusion.
    
    This service combines full-text search (SQLite FTS5) and vector search
    (LanceDB) results using Reciprocal Rank Fusion algorithm.
    """

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        vector_table: Optional[Any],
        sqlite_conn: Optional[Any]
    ):
        """Initialize SearchService with dependencies.
        
        Args:
            session: ONNX Runtime inference session for embedding.
            tokenizer: Hugging Face tokenizer for text processing.
            vector_table: LanceDB table for vector search.
            sqlite_conn: SQLite connection for full-text search.
        """
        self._session = session
        self._tokenizer = tokenizer
        self._vector_table = vector_table
        self._sqlite_conn = sqlite_conn

    def embed(self, text: str) -> np.ndarray:
        """Generate normalized embedding vector for text.
        
        Args:
            text: Input text to embed.
        
        Returns:
            L2-normalized 1024-dimensional embedding vector.
        """
        try:
            inputs = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
            outputs = self._session.run(None, inputs)
            embedding = outputs[0].mean(axis=1)[0]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
        except Exception as e:
            logging.error(f"Embed error: {e}")
            return np.zeros(1024, dtype=np.float32)

    def hybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
        rrf_k: int = DEFAULT_RRF_K
    ) -> List[Dict]:
        """Perform hybrid search combining FTS and vector search with RRF.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            threshold: Minimum similarity threshold for vector results.
            rrf_k: RRF constant (default 60).
        
        Returns:
            List of memory dicts with id, content, tags, note fields.
        """
        fetch_limit = top_k * FETCH_MULTIPLIER
        
        fts_results = self._search_fts(query, fetch_limit)
        vector_results = self._search_vector(query, fetch_limit, threshold)
        
        all_docs = {**fts_results, **vector_results}
        
        if not all_docs:
            return []
        
        fts_ranking = list(fts_results.keys())
        vector_ranking = list(vector_results.keys())
        
        ranked_lists = []
        if fts_ranking:
            ranked_lists.append(fts_ranking)
        if vector_ranking:
            ranked_lists.append(vector_ranking)
        
        if not ranked_lists:
            return []
        
        rrf_scores = fuse_rankings(ranked_lists, k=rrf_k)
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids[:top_k]:
            if doc_id in all_docs:
                results.append(all_docs[doc_id])
        
        return results

    def _search_fts(self, query: str, limit: int) -> Dict[str, Dict]:
        """Search SQLite FTS5 index.
        
        Args:
            query: Search query.
            limit: Maximum results to fetch.
        
        Returns:
            Dict mapping doc IDs to memory dicts.
        """
        results = {}
        
        if not self._sqlite_conn:
            return results
        
        try:
            cursor = self._sqlite_conn.execute(
                "SELECT id, content, tags, note FROM memories "
                "WHERE memories MATCH ? ORDER BY rank LIMIT ?",
                (f"{query}*", limit)
            )
            for row in cursor:
                doc_id = row[0]
                results[doc_id] = {
                    "id": doc_id,
                    "content": row[1],
                    "tags": row[2].split() if row[2] else [],
                    "note": row[3]
                }
        except Exception as e:
            logging.debug(f"FTS search error: {e}")
        
        return results

    def _search_vector(
        self,
        query: str,
        limit: int,
        threshold: float
    ) -> Dict[str, Dict]:
        """Search LanceDB vector index with similarity threshold.
        
        Args:
            query: Search query.
            limit: Maximum results to fetch.
            threshold: Minimum similarity score (0-1).
        
        Returns:
            Dict mapping doc IDs to memory dicts.
        """
        results = {}
        
        if not self._vector_table:
            return results
        
        try:
            query_vector = self.embed(query)
            hits = self._vector_table.search(query_vector).limit(limit).to_list()
            
            for hit in hits:
                distance = hit.get("_distance", 1.0)
                similarity = distance_to_similarity(distance)
                
                if similarity < threshold:
                    continue
                
                doc_id = hit["id"]
                results[doc_id] = {
                    "id": doc_id,
                    "content": hit["content"],
                    "tags": hit["tags"].split() if hit["tags"] else [],
                    "note": hit["note"]
                }
        except Exception as e:
            logging.debug(f"Vector search error: {e}")
        
        return results
