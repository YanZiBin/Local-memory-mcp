"""RRF (Reciprocal Rank Fusion) ranking algorithm implementation."""
from typing import Dict, List


def calculate_rrf_score(rank: int, k: int = 60) -> float:
    """Calculate RRF score for a given rank.
    
    Args:
        rank: The rank of the document (1-indexed).
        k: The RRF constant, default 60.
    
    Returns:
        The RRF score: 1 / (k + rank).
    """
    return 1.0 / (k + rank)


def fuse_rankings(ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: A list of ranked document ID lists. Each inner list
            contains document IDs ordered by rank (best first).
        k: The RRF constant, default 60.
    
    Returns:
        A dictionary mapping document IDs to their aggregated RRF scores.
    """
    scores: Dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            rrf_score = calculate_rrf_score(rank, k)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
    
    return scores


def distance_to_similarity(distance: float) -> float:
    """Convert LanceDB distance to similarity score.
    
    For normalized vectors using L2 distance, similarity = 1 - distance.
    The result is clamped to [0.0, 1.0].
    
    Args:
        distance: The distance from LanceDB search result.
    
    Returns:
        Similarity score in range [0.0, 1.0].
    """
    return max(0.0, 1.0 - distance)
