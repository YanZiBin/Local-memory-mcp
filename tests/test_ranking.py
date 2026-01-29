"""Unit tests for RRF ranking algorithm."""
import pytest
from ranking import calculate_rrf_score, fuse_rankings, distance_to_similarity


class TestCalculateRrfScore:
    """Tests for calculate_rrf_score function."""

    def test_rank_1_with_default_k(self):
        """Rank 1 with default k=60 should return 1/61."""
        score = calculate_rrf_score(rank=1, k=60)
        assert score == pytest.approx(1 / 61)

    def test_rank_1_with_custom_k(self):
        """Rank 1 with k=10 should return 1/11."""
        score = calculate_rrf_score(rank=1, k=10)
        assert score == pytest.approx(1 / 11)

    def test_higher_rank_lower_score(self):
        """Higher rank should result in lower score."""
        score_rank_1 = calculate_rrf_score(rank=1)
        score_rank_5 = calculate_rrf_score(rank=5)
        score_rank_10 = calculate_rrf_score(rank=10)
        assert score_rank_1 > score_rank_5 > score_rank_10


class TestFuseRankings:
    """Tests for fuse_rankings function."""

    def test_single_source(self):
        """Single source should return RRF scores for each doc."""
        ranked_lists = [["doc_a", "doc_b", "doc_c"]]
        scores = fuse_rankings(ranked_lists, k=60)
        
        assert "doc_a" in scores
        assert "doc_b" in scores
        assert "doc_c" in scores
        assert scores["doc_a"] > scores["doc_b"] > scores["doc_c"]

    def test_two_sources_same_order(self):
        """Two sources with same order should double the scores."""
        ranked_lists = [
            ["doc_a", "doc_b"],
            ["doc_a", "doc_b"]
        ]
        scores = fuse_rankings(ranked_lists, k=60)
        
        expected_doc_a = 2 * (1 / 61)
        assert scores["doc_a"] == pytest.approx(expected_doc_a)

    def test_two_sources_different_order(self):
        """Document appearing in both sources at different ranks."""
        ranked_lists = [
            ["doc_a", "doc_b", "doc_c"],
            ["doc_c", "doc_a", "doc_b"]
        ]
        scores = fuse_rankings(ranked_lists, k=60)
        
        # doc_a: rank 1 in source 1 + rank 2 in source 2
        expected_doc_a = 1/61 + 1/62
        assert scores["doc_a"] == pytest.approx(expected_doc_a)
        
        # doc_c: rank 3 in source 1 + rank 1 in source 2
        expected_doc_c = 1/63 + 1/61
        assert scores["doc_c"] == pytest.approx(expected_doc_c)

    def test_document_in_one_source_only(self):
        """Document appearing in only one source."""
        ranked_lists = [
            ["doc_a", "doc_b"],
            ["doc_c", "doc_d"]
        ]
        scores = fuse_rankings(ranked_lists, k=60)
        
        assert scores["doc_a"] == pytest.approx(1/61)
        assert scores["doc_c"] == pytest.approx(1/61)

    def test_empty_sources(self):
        """Empty ranked lists should return empty scores."""
        scores = fuse_rankings([], k=60)
        assert scores == {}


class TestDistanceToSimilarity:
    """Tests for distance_to_similarity function."""

    def test_zero_distance_max_similarity(self):
        """Zero distance should return similarity of 1.0."""
        similarity = distance_to_similarity(0.0)
        assert similarity == 1.0

    def test_distance_one_zero_similarity(self):
        """Distance of 1.0 should return similarity of 0.0."""
        similarity = distance_to_similarity(1.0)
        assert similarity == 0.0

    def test_partial_distance(self):
        """Distance of 0.3 should return similarity of 0.7."""
        similarity = distance_to_similarity(0.3)
        assert similarity == pytest.approx(0.7)

    def test_negative_similarity_clamped(self):
        """Distance > 1 should clamp similarity to 0."""
        similarity = distance_to_similarity(1.5)
        assert similarity == 0.0
