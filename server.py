import os
import logging
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastmcp import FastMCP
import numpy as np

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 1. Initialize resources eagerly ---
logging.info("Initializing resources... Please wait for TensorRT/CUDA loading...")

import lancedb
import sqlite3
import onnxruntime as ort
from transformers import AutoTokenizer

# Tokenizer
model_dir = os.path.join(os.path.dirname(__file__), "bge-m3-onnx")
logging.info(f"Loading tokenizer from {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ONNX Session
onnx_path = os.path.join(model_dir, "sentence_transformers.onnx")
logging.info(f"Loading ONNX Model (This may take a while): {onnx_path}")

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.log_severity_level = 1  # Show warnings

session = ort.InferenceSession(
    onnx_path,
    sess_options=session_options,
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)
logging.info(">>> ONNX Model Loaded Successfully! <<<")

# LanceDB
logging.info("Connecting to LanceDB...")
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_db")
db = lancedb.connect(db_path)
try:
    vector_table = db.open_table("memories")
except Exception:
    vector_table = db.create_table(
        "memories",
        data=[{
            "vector": np.zeros(1024, dtype=np.float32),
            "id": "dummy",
            "content": "",
            "tags": "",
            "note": ""
        }],
        mode="create"
    )

# SQLite
logging.info("Connecting to SQLite...")
sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")
conn = sqlite3.connect(sqlite_path, check_same_thread=False)
conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
        id, content, tags, note, tokenize='unicode61'
    )
""")
conn.commit()

logging.info(">>> All Resources Ready! Server is starting... <<<")

# Initialize SearchService
from search_engine import SearchService

search_service = SearchService(
    session=session,
    tokenizer=tokenizer,
    vector_table=vector_table,
    sqlite_conn=conn
)

# --- Define app ---
app = FastMCP("Project Memory Bank (SSE Mode)")


def _tags_to_string(tags: List[str]) -> str:
    """Convert tags into the storage format used by SQLite and LanceDB."""
    return " ".join(tags)


def _memory_from_row(row: Any) -> Dict[str, Any]:
    """Convert a SQLite row into a memory payload."""
    return {
        "id": row[0],
        "content": row[1],
        "tags": row[2].split() if row[2] else [],
        "note": row[3]
    }


def _build_memory_response(
    memory: Dict[str, Any],
    message: str,
    updated_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Build a structured success response for a memory record."""
    return {
        "id": memory["id"],
        "content": memory["content"],
        "tags": memory["tags"],
        "note": memory["note"],
        "message": message,
        "updated_fields": updated_fields or []
    }


def _build_error_response(message: str, memory_id: Optional[str] = None) -> Dict[str, Any]:
    """Build a structured error response."""
    response = {
        "message": message,
        "updated_fields": []
    }
    if memory_id:
        response["id"] = memory_id
    return response


def _get_memory_by_id(memory_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single memory record by ID from SQLite."""
    if conn is None:
        return None

    cursor = conn.execute(
        "SELECT id, content, tags, note FROM memories WHERE id = ?",
        (memory_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return None

    return _memory_from_row(row)


def _validate_update_request(
    memory_id: str,
    content: Optional[str],
    tags: Optional[List[str]],
    note: Optional[str]
) -> Dict[str, Any]:
    """Validate inputs for update_memory and normalize tags."""
    if not memory_id or not memory_id.strip():
        return {"error": "memory_id is required."}

    if content is None and tags is None and note is None:
        return {"error": "At least one of content, tags, or note must be provided."}

    if content is not None and not content.strip():
        return {"error": "content cannot be empty. Clearing fields is not supported."}

    if note is not None and not note.strip():
        return {"error": "note cannot be empty. Clearing fields is not supported."}

    normalized_tags = None
    if tags is not None:
        if len(tags) == 0:
            return {"error": "tags cannot be empty. Clearing fields is not supported."}

        normalized_tags = []
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                return {"error": "tags cannot contain empty values."}
            normalized_tags.append(tag.strip())

    return {"normalized_tags": normalized_tags}


def _prepare_updated_memory(
    current_memory: Dict[str, Any],
    content: Optional[str],
    tags: Optional[List[str]],
    note: Optional[str]
) -> Dict[str, Any]:
    """Merge partial updates into the current record."""
    updated_memory = {
        "id": current_memory["id"],
        "content": current_memory["content"],
        "tags": list(current_memory["tags"]),
        "note": current_memory["note"]
    }
    updated_fields = []

    if content is not None and content != current_memory["content"]:
        updated_memory["content"] = content
        updated_fields.append("content")

    if tags is not None and tags != current_memory["tags"]:
        updated_memory["tags"] = list(tags)
        updated_fields.append("tags")

    if note is not None and note != current_memory["note"]:
        updated_memory["note"] = note
        updated_fields.append("note")

    return {
        "memory": updated_memory,
        "updated_fields": updated_fields
    }


def _build_vector_payload(memory: Dict[str, Any], vector: np.ndarray) -> Dict[str, Any]:
    """Build the LanceDB payload for a memory record."""
    return {
        "vector": vector,
        "id": memory["id"],
        "content": memory["content"],
        "tags": _tags_to_string(memory["tags"]),
        "note": memory["note"]
    }


def _delete_vector_memory(memory_id: str) -> None:
    """Delete a memory from LanceDB."""
    if vector_table is None:
        raise RuntimeError("Vector table is not available.")

    escaped_id = memory_id.replace("'", "''")
    vector_table.delete(f"id = '{escaped_id}'")


def _add_vector_memory(memory: Dict[str, Any], vector: np.ndarray) -> None:
    """Insert a memory into LanceDB."""
    if vector_table is None:
        raise RuntimeError("Vector table is not available.")

    vector_table.add([_build_vector_payload(memory, vector)])


@app.tool("save_memory")
def save_memory(
    content: str,
    tags: List[str] = None,
    note: str = "",
    skip_duplicate_check: bool = False
) -> Dict[str, Any]:
    """Save a project memory with optional duplicate detection.
    
    Args:
        content: The memory content to save.
        tags: Optional list of tags.
        note: Optional note field.
        skip_duplicate_check: If True, skip duplicate detection (default False).
    
    Returns:
        A dict with status, memory_id, and optional duplicate warnings.
    """
    logging.info(f"Tool called: save_memory | Content: {content[:20]}...")
    try:
        # Step 1: Check for duplicates (unless skipped)
        if not skip_duplicate_check:
            similar_memories = search_service.find_similar(content)
            if similar_memories:
                # Found potential duplicates
                return {
                    "status": "duplicate_detected",
                    "message": "Potential duplicate memories found. Review before saving.",
                    "similar_memories": similar_memories,
                    "action_required": "Review similar memories and decide whether to save anyway by setting skip_duplicate_check=true"
                }
        
        # Step 2: Save the memory
        memory_id = str(uuid.uuid4())
        tags_str = _tags_to_string(tags or [])

        # SQLite
        if conn:
            conn.execute(
                "INSERT INTO memories(id, content, tags, note) VALUES (?, ?, ?, ?)",
                (memory_id, content, tags_str, note)
            )
            conn.commit()

        # LanceDB
        vector = search_service.embed(content)
        if vector_table:
            vector_table.add([{
                "vector": vector,
                "id": memory_id,
                "content": content,
                "tags": tags_str,
                "note": note
            }])

        logging.info(f"Success! Memory saved: {memory_id}")
        return {
            "status": "saved",
            "memory_id": memory_id,
            "message": "Memory saved successfully.",
            "content": content,
            "tags": tags or [],
            "note": note
        }
    except Exception as e:
        logging.error(f"Error saving memory: {e}")
        return {
            "status": "error",
            "message": f"Error saving memory: {e}"
        }


@app.tool("update_memory")
def update_memory(
    memory_id: str,
    content: Optional[str] = None,
    tags: Optional[List[str]] = None,
    note: Optional[str] = None
) -> Dict[str, Any]:
    """Update a single memory by ID."""
    logging.info(f"Tool called: update_memory | ID: {memory_id}")

    normalized_memory_id = memory_id.strip() if memory_id else ""
    validation = _validate_update_request(memory_id, content, tags, note)
    if "error" in validation:
        return _build_error_response(validation["error"], normalized_memory_id or None)

    normalized_tags = validation["normalized_tags"]

    try:
        current_memory = _get_memory_by_id(normalized_memory_id)
        if current_memory is None:
            return _build_error_response(
                f"Memory {normalized_memory_id} not found.",
                normalized_memory_id
            )

        update_payload = _prepare_updated_memory(
            current_memory=current_memory,
            content=content,
            tags=normalized_tags if tags is not None else None,
            note=note
        )
        updated_memory = update_payload["memory"]
        updated_fields = update_payload["updated_fields"]

        if not updated_fields:
            return _build_memory_response(
                current_memory,
                "No changes applied to memory.",
                []
            )

        if conn is None:
            raise RuntimeError("SQLite connection is not available.")

        updated_tags_str = _tags_to_string(updated_memory["tags"])
        new_vector = search_service.embed(updated_memory["content"])
        old_vector = search_service.embed(current_memory["content"])
        vector_deleted = False

        conn.execute("BEGIN")
        conn.execute(
            "UPDATE memories SET content = ?, tags = ?, note = ? WHERE id = ?",
            (
                updated_memory["content"],
                updated_tags_str,
                updated_memory["note"],
                normalized_memory_id
            )
        )

        try:
            _delete_vector_memory(normalized_memory_id)
            vector_deleted = True
            _add_vector_memory(updated_memory, new_vector)
        except Exception as vector_error:
            conn.rollback()

            if vector_deleted:
                try:
                    _add_vector_memory(current_memory, old_vector)
                except Exception as restore_error:
                    logging.error(
                        "Failed to restore vector record for %s after update error: %s",
                        normalized_memory_id,
                        restore_error
                    )
                    return _build_error_response(
                        "Error updating memory: vector sync failed and previous vector "
                        f"could not be restored ({restore_error})",
                        normalized_memory_id
                    )

            logging.error(
                "Vector sync failed while updating %s: %s",
                normalized_memory_id,
                vector_error
            )
            return _build_error_response(
                f"Error updating memory: vector sync failed ({vector_error})",
                normalized_memory_id
            )

        conn.commit()
        logging.info("Memory %s updated successfully.", normalized_memory_id)
        return _build_memory_response(
            updated_memory,
            "Memory updated successfully.",
            updated_fields
        )
    except Exception as e:
        try:
            if conn:
                conn.rollback()
        except Exception:
            logging.debug("Rollback skipped after update error.")

        logging.error(f"Update error: {e}")
        return _build_error_response(
            f"Error updating memory: {e}",
            normalized_memory_id or None
        )


@app.tool("search_memory")
def search_memory(query: str, top_k: int = 5) -> List[Dict]:
    """Search memories."""
    logging.info(f"Tool called: search_memory | Query: {query}")
    try:
        results = search_service.hybrid_search(query, top_k=top_k)
        logging.info(f"Found {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"Search error: {e}")
        return []


@app.tool("list_memories")
def list_memories(limit: int = 10, offset: int = 0) -> List[Dict]:
    """List recently saved memories with pagination."""
    logging.info(f"Tool called: list_memories | Limit: {limit}, Offset: {offset}")
    try:
        results = []
        if conn:
            cursor = conn.execute(
                "SELECT id, content, tags, note FROM memories ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            for row in cursor:
                results.append(_memory_from_row(row))
        return results
    except Exception as e:
        logging.error(f"List memories error: {e}")
        return []


@app.tool("delete_memory")
def delete_memory(memory_id: str) -> str:
    """Delete a memory permanently by ID."""
    logging.info(f"Tool called: delete_memory | ID: {memory_id}")
    try:
        # 1. Delete SQLite record
        if conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            logging.info("Deleted from SQLite")

        # 2. Delete LanceDB record
        if vector_table:
            try:
                vector_table.delete(f"id = '{memory_id}'")
                logging.info("Deleted from LanceDB")
            except Exception as le:
                logging.warning(f"LanceDB delete warning (might not exist): {le}")

        return f"Memory {memory_id} deleted successfully."
    except Exception as e:
        logging.error(f"Delete error: {e}")
        return f"Error deleting memory: {e}"


if __name__ == "__main__":
    logging.info("Starting SSE Server on port 8000...")
    app.run(transport="sse", host="0.0.0.0", port=8000)
