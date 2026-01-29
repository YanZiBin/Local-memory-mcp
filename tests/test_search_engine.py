"""Unit tests for SearchService."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


class TestSearchServiceEmbed:
    """Tests for SearchService.embed method."""

    def test_embed_returns_numpy_array(self):
        """Embed should return a numpy ndarray."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=None,
            sqlite_conn=None
        )
        
        result = service.embed("test text")
        assert isinstance(result, np.ndarray)

    def test_embed_returns_normalized_vector(self):
        """Embed output should be L2-normalized."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=None,
            sqlite_conn=None
        )
        
        result = service.embed("test text")
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_embed_returns_correct_dimension(self):
        """Embed should return 1024-dimensional vector."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=None,
            sqlite_conn=None
        )
        
        result = service.embed("test text")
        assert result.shape == (1024,)


class TestSearchServiceHybridSearch:
    """Tests for SearchService.hybrid_search method."""

    def test_hybrid_search_returns_list(self):
        """Hybrid search should return a list."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        mock_conn = Mock()
        mock_conn.execute.return_value = []
        
        mock_table = Mock()
        mock_search = Mock()
        mock_search.limit.return_value.to_list.return_value = []
        mock_table.search.return_value = mock_search
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=mock_table,
            sqlite_conn=mock_conn
        )
        
        result = service.hybrid_search("test query")
        assert isinstance(result, list)

    def test_hybrid_search_applies_threshold(self):
        """Results below similarity threshold should be filtered out."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        mock_conn = Mock()
        mock_conn.execute.return_value = []
        
        # Vector results with varying distances
        mock_table = Mock()
        mock_search = Mock()
        mock_search.limit.return_value.to_list.return_value = [
            {"id": "doc1", "content": "close match", "tags": "", "note": "", "_distance": 0.1},
            {"id": "doc2", "content": "far match", "tags": "", "note": "", "_distance": 0.9},
        ]
        mock_table.search.return_value = mock_search
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=mock_table,
            sqlite_conn=mock_conn
        )
        
        result = service.hybrid_search("test query", threshold=0.5)
        
        result_ids = [r["id"] for r in result]
        assert "doc1" in result_ids
        assert "doc2" not in result_ids

    def test_hybrid_search_output_format(self):
        """Results should have id, content, tags, note fields."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        mock_conn = Mock()
        mock_conn.execute.return_value = [
            ("id1", "content1", "tag1 tag2", "note1")
        ]
        
        mock_table = Mock()
        mock_search = Mock()
        mock_search.limit.return_value.to_list.return_value = []
        mock_table.search.return_value = mock_search
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=mock_table,
            sqlite_conn=mock_conn
        )
        
        result = service.hybrid_search("test query")
        
        assert len(result) > 0
        item = result[0]
        assert "id" in item
        assert "content" in item
        assert "tags" in item
        assert "note" in item
        assert isinstance(item["tags"], list)

    def test_hybrid_search_respects_top_k(self):
        """Should return at most top_k results."""
        from search_engine import SearchService
        
        mock_session = Mock()
        mock_session.run.return_value = [np.random.randn(1, 10, 1024).astype(np.float32)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        
        mock_conn = Mock()
        mock_conn.execute.return_value = [
            (f"id{i}", f"content{i}", "", "") for i in range(10)
        ]
        
        mock_table = Mock()
        mock_search = Mock()
        mock_search.limit.return_value.to_list.return_value = []
        mock_table.search.return_value = mock_search
        
        service = SearchService(
            session=mock_session,
            tokenizer=mock_tokenizer,
            vector_table=mock_table,
            sqlite_conn=mock_conn
        )
        
        result = service.hybrid_search("test query", top_k=3)
        assert len(result) <= 3
