"""Tests for duplicate detection in save_memory."""
import importlib
import sqlite3
from types import ModuleType

import numpy as np
import pytest


class FakeFastMCP:
    """Minimal FastMCP stub used to import server.py in tests."""

    def __init__(self, name: str):
        self.name = name

    def tool(self, _tool_name: str):
        def decorator(func):
            return func
        return decorator

    def run(self, *args, **kwargs):
        return None


class FakeVectorSearch:
    """Chainable LanceDB search stub."""

    def __init__(self, records):
        self._records = records
        self._limit = None

    def limit(self, limit: int):
        self._limit = limit
        return self

    def to_list(self):
        records = list(self._records.values())
        if self._limit is not None:
            records = records[:self._limit]
        return [{**record, "_distance": 0.1} for record in records]


class FakeVectorTable:
    """In-memory LanceDB table stub with failure injection hooks."""

    def __init__(self):
        self.records = {}
        self.add_calls = []
        self.delete_calls = []
        self.fail_on_delete = False
        self.add_failures = []

    def add(self, items):
        self.add_calls.append(items)
        should_fail = self.add_failures.pop(0) if self.add_failures else False
        if should_fail:
            raise RuntimeError("vector add failed")

        for item in items:
            self.records[item["id"]] = dict(item)

    def delete(self, filter_expression: str):
        self.delete_calls.append(filter_expression)
        if self.fail_on_delete:
            raise RuntimeError("vector delete failed")

        memory_id = filter_expression.split("'")[1]
        self.records.pop(memory_id, None)

    def search(self, _query_vector):
        return FakeVectorSearch(self.records)


class FakeLanceDb:
    """Connection stub that always returns the shared fake table."""

    def __init__(self, table: FakeVectorTable):
        self._table = table

    def open_table(self, _name: str):
        return self._table

    def create_table(self, _name: str, data=None, mode=None):
        if data:
            for item in data:
                self._table.records[item["id"]] = dict(item)
        return self._table


class FakeSessionOptions:
    """Minimal onnxruntime.SessionOptions replacement."""

    def __init__(self):
        self.graph_optimization_level = None
        self.log_severity_level = None


class FakeInferenceSession:
    """Minimal onnxruntime.InferenceSession replacement."""

    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, *_args, **_kwargs):
        return [np.ones((1, 10, 1024), dtype=np.float32)]


class FakeTokenizer:
    """Tokenizer stub used by SearchService during import."""

    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }


@pytest.fixture
def server_module(monkeypatch):
    """Import server.py with lightweight fake dependencies."""
    real_sqlite_connect = sqlite3.connect
    fake_vector_table = FakeVectorTable()

    fastmcp_module = ModuleType("fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP

    lancedb_module = ModuleType("lancedb")
    lancedb_module.connect = lambda _path: FakeLanceDb(fake_vector_table)

    ort_module = ModuleType("onnxruntime")
    ort_module.SessionOptions = FakeSessionOptions
    ort_module.InferenceSession = FakeInferenceSession
    ort_module.GraphOptimizationLevel = type(
        "GraphOptimizationLevel",
        (),
        {"ORT_ENABLE_ALL": 1}
    )

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = type(
        "AutoTokenizer",
        (),
        {"from_pretrained": staticmethod(lambda _path: FakeTokenizer())}
    )

    uvicorn_module = ModuleType("uvicorn")

    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_module)
    monkeypatch.setitem(sys.modules, "lancedb", lancedb_module)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *_args, **_kwargs: real_sqlite_connect(":memory:", check_same_thread=False)
    )

    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    module.vector_table = fake_vector_table
    module.search_service._vector_table = fake_vector_table

    try:
        yield module
    finally:
        module.conn.close()
        sys.modules.pop("server", None)


def seed_memory(module, memory_id="memory-1", content="old content", tags=None, note="old note"):
    """Insert a memory record into the fake SQLite and vector stores."""
    tags = tags or ["old", "tag"]
    tags_str = " ".join(tags)
    module.conn.execute(
        "INSERT INTO memories(id, content, tags, note) VALUES (?, ?, ?, ?)",
        (memory_id, content, tags_str, note)
    )
    module.conn.commit()
    module.vector_table.records[memory_id] = {
        "id": memory_id,
        "content": content,
        "tags": tags_str,
        "note": note,
        "vector": np.zeros(1024, dtype=np.float32)
    }
    return memory_id


def test_save_memory_detects_duplicate(server_module, monkeypatch):
    """Saving duplicate content should be detected and rejected."""
    seed_memory(server_module, content="My project uses Python 3.10")
    
    # Mock find_similar to return a duplicate
    def mock_find_similar(content, threshold=0.85, top_k=5):
        if "Python 3.10" in content:
            return [{
                "id": "memory-1",
                "content": "My project uses Python 3.10",
                "tags": ["old", "tag"],
                "note": "old note",
                "similarity": 0.95
            }]
        return []
    
    monkeypatch.setattr(server_module.search_service, "find_similar", mock_find_similar)
    
    result = server_module.save_memory(content="My project uses Python 3.10")
    
    assert result["status"] == "duplicate_detected"
    assert "similar_memories" in result
    assert len(result["similar_memories"]) == 1
    assert result["similar_memories"][0]["similarity"] == 0.95


def test_save_memory_allows_unique_content(server_module, monkeypatch):
    """Saving unique content should succeed."""
    # Mock find_similar to return no duplicates
    monkeypatch.setattr(
        server_module.search_service,
        "find_similar",
        lambda content, threshold=0.85, top_k=5: []
    )
    monkeypatch.setattr(
        server_module.search_service,
        "embed",
        lambda text: np.zeros(1024, dtype=np.float32)
    )
    
    result = server_module.save_memory(content="This is unique content")
    
    assert result["status"] == "saved"
    assert "memory_id" in result
    assert result["content"] == "This is unique content"


def test_save_memory_skip_duplicate_check(server_module, monkeypatch):
    """Setting skip_duplicate_check=True should bypass duplicate detection."""
    seed_memory(server_module, content="My project uses Python 3.10")
    
    # Mock find_similar to return a duplicate
    def mock_find_similar(content, threshold=0.85, top_k=5):
        if "Python 3.10" in content:
            return [{
                "id": "memory-1",
                "content": "My project uses Python 3.10",
                "tags": ["old", "tag"],
                "note": "old note",
                "similarity": 0.95
            }]
        return []
    
    monkeypatch.setattr(server_module.search_service, "find_similar", mock_find_similar)
    monkeypatch.setattr(
        server_module.search_service,
        "embed",
        lambda text: np.zeros(1024, dtype=np.float32)
    )
    
    # Should save anyway because skip_duplicate_check=True
    result = server_module.save_memory(
        content="My project uses Python 3.10",
        skip_duplicate_check=True
    )
    
    assert result["status"] == "saved"
    assert "memory_id" in result


def test_save_memory_returns_error_on_exception(server_module, monkeypatch):
    """Saving should return error status on exception."""
    monkeypatch.setattr(
        server_module.search_service,
        "find_similar",
        lambda content, threshold=0.85, top_k=5: []
    )
    monkeypatch.setattr(
        server_module.search_service,
        "embed",
        lambda text: (_ for _ in ()).throw(RuntimeError("embed failed"))
    )
    
    result = server_module.save_memory(content="test content")
    
    assert result["status"] == "error"
    assert "embed failed" in result["message"]


def test_find_similar_integration(server_module, monkeypatch):
    """Test find_similar method directly."""
    seed_memory(
        server_module,
        content="Project uses Python 3.10 and FastMCP",
        tags=["project", "python"]
    )
    
    # Mock embed to return consistent vectors
    monkeypatch.setattr(
        server_module.search_service,
        "embed",
        lambda text: np.zeros(1024, dtype=np.float32)
    )
    
    results = server_module.search_service.find_similar(
        "Project uses Python 3.10 and FastMCP",
        threshold=0.7
    )
    
    assert len(results) >= 1
    assert results[0]["content"] == "Project uses Python 3.10 and FastMCP"
    assert "similarity" in results[0]
