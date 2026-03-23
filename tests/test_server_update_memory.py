"""Tests for server-level memory update behavior."""
import importlib
import sqlite3
import sys
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


def get_memory_row(module, memory_id):
    """Fetch a memory row directly from SQLite."""
    cursor = module.conn.execute(
        "SELECT id, content, tags, note FROM memories WHERE id = ?",
        (memory_id,)
    )
    return cursor.fetchone()


def test_update_memory_updates_content_and_reindexes_vector(server_module, monkeypatch):
    """Updating content should rewrite SQLite and LanceDB immediately."""
    memory_id = seed_memory(server_module)
    embed_calls = []

    def fake_embed(text):
        embed_calls.append(text)
        return np.full(1024, len(text), dtype=np.float32)

    monkeypatch.setattr(server_module.search_service, "embed", fake_embed)

    result = server_module.update_memory(memory_id=memory_id, content="new content")

    assert result["message"] == "Memory updated successfully."
    assert result["updated_fields"] == ["content"]
    assert result["content"] == "new content"
    assert "new content" in embed_calls

    row = get_memory_row(server_module, memory_id)
    assert row[1] == "new content"
    assert server_module.vector_table.records[memory_id]["content"] == "new content"


def test_update_memory_updates_tags_only(server_module):
    """Tag-only updates should keep content and note unchanged."""
    memory_id = seed_memory(server_module, tags=["one", "two"])

    result = server_module.update_memory(memory_id=memory_id, tags=["new", "tags"])

    assert result["updated_fields"] == ["tags"]
    assert result["content"] == "old content"
    assert result["note"] == "old note"

    row = get_memory_row(server_module, memory_id)
    assert row[2] == "new tags"
    assert server_module.vector_table.records[memory_id]["tags"] == "new tags"


def test_update_memory_updates_note_only(server_module):
    """Note-only updates should preserve other fields."""
    memory_id = seed_memory(server_module)

    result = server_module.update_memory(memory_id=memory_id, note="updated note")

    assert result["updated_fields"] == ["note"]
    assert result["note"] == "updated note"

    row = get_memory_row(server_module, memory_id)
    assert row[3] == "updated note"
    assert server_module.vector_table.records[memory_id]["note"] == "updated note"


def test_update_memory_updates_multiple_fields(server_module):
    """Multiple fields should update in one call."""
    memory_id = seed_memory(server_module)

    result = server_module.update_memory(
        memory_id=memory_id,
        content="project memory updated",
        tags=["project", "memory"],
        note="fresh note"
    )

    assert result["updated_fields"] == ["content", "tags", "note"]
    assert result["content"] == "project memory updated"
    assert result["tags"] == ["project", "memory"]
    assert result["note"] == "fresh note"

    row = get_memory_row(server_module, memory_id)
    assert row[1:] == ("project memory updated", "project memory", "fresh note")


def test_update_memory_rejects_missing_update_fields(server_module):
    """At least one mutable field must be provided."""
    memory_id = seed_memory(server_module)

    result = server_module.update_memory(memory_id=memory_id)

    assert result["message"] == "At least one of content, tags, or note must be provided."
    assert result["updated_fields"] == []


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"content": ""}, "content cannot be empty. Clearing fields is not supported."),
        ({"note": ""}, "note cannot be empty. Clearing fields is not supported."),
        ({"tags": []}, "tags cannot be empty. Clearing fields is not supported.")
    ]
)
def test_update_memory_rejects_clearing_fields(server_module, kwargs, expected_message):
    """Clearing a field explicitly is not supported."""
    memory_id = seed_memory(server_module)

    result = server_module.update_memory(memory_id=memory_id, **kwargs)

    assert result["message"] == expected_message
    assert result["updated_fields"] == []


def test_update_memory_returns_not_found_for_unknown_id(server_module):
    """Unknown IDs should produce a clear error response."""
    result = server_module.update_memory(memory_id="missing-id", content="new value")

    assert result["message"] == "Memory missing-id not found."
    assert result["id"] == "missing-id"


def test_update_memory_returns_no_change_without_writing(server_module):
    """No-op updates should not touch either storage backend."""
    memory_id = seed_memory(
        server_module,
        content="same content",
        tags=["same"],
        note="same note"
    )

    result = server_module.update_memory(memory_id=memory_id, content="same content")

    assert result["message"] == "No changes applied to memory."
    assert result["updated_fields"] == []
    assert server_module.vector_table.delete_calls == []
    assert len(server_module.vector_table.add_calls) == 0


def test_update_memory_rolls_back_sqlite_when_vector_sync_fails(server_module):
    """SQLite should be rolled back when LanceDB add fails."""
    memory_id = seed_memory(server_module, content="original content")
    server_module.vector_table.add_failures = [True, False]

    result = server_module.update_memory(memory_id=memory_id, content="new content")

    assert "vector sync failed" in result["message"]
    row = get_memory_row(server_module, memory_id)
    assert row[1] == "original content"
    assert server_module.vector_table.records[memory_id]["content"] == "original content"


def test_update_memory_reports_restore_failure(server_module):
    """If vector restoration fails, the error should say so clearly."""
    memory_id = seed_memory(server_module, content="original content")
    server_module.vector_table.add_failures = [True, True]

    result = server_module.update_memory(memory_id=memory_id, content="new content")

    assert "could not be restored" in result["message"]
    row = get_memory_row(server_module, memory_id)
    assert row[1] == "original content"
    assert memory_id not in server_module.vector_table.records


def test_updated_memory_is_visible_in_list_and_search(server_module):
    """Updated content should be visible to list and FTS search immediately."""
    memory_id = seed_memory(server_module, content="legacy memory content")
    update_result = server_module.update_memory(
        memory_id=memory_id,
        content="updated project memory"
    )

    assert update_result["message"] == "Memory updated successfully."

    server_module.vector_table = None
    server_module.search_service._vector_table = None

    listed = server_module.list_memories(limit=5)
    searched = server_module.search_memory("updated", top_k=5)

    assert any(
        item["id"] == memory_id and item["content"] == "updated project memory"
        for item in listed
    )
    assert any(
        item["id"] == memory_id and item["content"] == "updated project memory"
        for item in searched
    )


def test_save_and_delete_memory_still_work(server_module, monkeypatch):
    """Existing save/delete tools should still behave after the update feature lands."""
    monkeypatch.setattr(
        server_module.search_service,
        "embed",
        lambda _text: np.zeros(1024, dtype=np.float32)
    )

    save_result = server_module.save_memory("saved content", tags=["keep"], note="note")
    memory_id = save_result.split(": ", 1)[1]

    assert get_memory_row(server_module, memory_id)[1] == "saved content"
    assert memory_id in server_module.vector_table.records

    delete_result = server_module.delete_memory(memory_id)

    assert delete_result == f"Memory {memory_id} deleted successfully."
    assert get_memory_row(server_module, memory_id) is None
    assert memory_id not in server_module.vector_table.records
