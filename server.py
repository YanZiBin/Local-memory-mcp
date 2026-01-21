import os
import logging
import uuid
from typing import List, Dict
import uvicorn
from fastmcp import FastMCP
import numpy as np

# --- 配置日志到控制台 (方便你看) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 1. 初始化资源 (Eager Loading - 既然手动启动，就直接加载!) ---
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
session_options.log_severity_level = 1 # Show warnings

session = ort.InferenceSession(
    onnx_path,
    sess_options=session_options,
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
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
        data=[{"vector": np.zeros(1024, dtype=np.float32), "id": "dummy", "content": "", "tags": "", "note": ""}],
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


def embed(text: str) -> np.ndarray:
    try:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()} 
        outputs = session.run(None, inputs)
        embedding = outputs[0].mean(axis=1)[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
    except Exception as e:
        logging.error(f"Embed error: {e}")
        return np.zeros(1024, dtype=np.float32)

# --- 定义 App (显式禁用 Redis 逻辑已在头部通过 env 实现) ---
app = FastMCP(
    "Project Memory Bank (SSE Mode)"
)

@app.tool("save_memory")
def save_memory(content: str, tags: List[str] = None, note: str = "") -> str:
    """保存一条项目记忆"""
    logging.info(f"Tool called: save_memory | Content: {content[:20]}...")
    try:
        memory_id = str(uuid.uuid4())
        tags_str = " ".join(tags or [])

        # SQLite
        if conn:
            conn.execute(
                "INSERT INTO memories(id, content, tags, note) VALUES (?, ?, ?, ?)",
                (memory_id, content, tags_str, note)
            )
            conn.commit()

        # LanceDB
        vector = embed(content)
        if vector_table:
            vector_table.add([{
                "vector": vector,
                "id": memory_id,
                "content": content,
                "tags": tags_str,
                "note": note
            }])
            
        logging.info(f"Success! Memory saved: {memory_id}")
        return f"Memory saved with id: {memory_id}"
    except Exception as e:
        logging.error(f"Error saving memory: {e}")
        return f"Error: {e}"

@app.tool("search_memory")
def search_memory(query: str, top_k: int = 5) -> List[Dict]:
    """搜索记忆"""
    logging.info(f"Tool called: search_memory | Query: {query}")
    try:
        results = {}

        # SQLite Search
        if conn:
            try:
                for row in conn.execute(
                    "SELECT id, content, tags, note FROM memories WHERE memories MATCH ? ORDER BY rank LIMIT ?",
                    (f"{query}*", top_k * 2)
                ):
                    mid = row[0]
                    results[mid] = {"id": mid, "content": row[1], "tags": row[2].split(), "note": row[3]}
            except: pass

        # LanceDB Search
        if vector_table:
            try:
                query_vec = embed(query)
                vec_hits = vector_table.search(query_vec).limit(top_k * 2).to_list()
                for hit in vec_hits:
                    mid = hit["id"]
                    results[mid] = {
                        "id": mid,
                        "content": hit["content"],
                        "tags": hit["tags"].split(),
                        "note": hit["note"]
                    }
            except: pass
        
        final_results = list(results.values())[:top_k]
        logging.info(f"Found {len(final_results)} results.")
        return final_results
    except Exception as e:
        logging.error(f"Search error: {e}")
        return []

@app.tool("list_memories")
def list_memories(limit: int = 10, offset: int = 0) -> List[Dict]:
    """列出最近保存的记忆 (支持分页，默认返回最新的10条)"""
    logging.info(f"Tool called: list_memories | Limit: {limit}, Offset: {offset}")
    try:
        results = []
        if conn:
            # 按 rowid 倒序排列，这样能看到最新的记忆
            cursor = conn.execute(
                "SELECT id, content, tags, note FROM memories ORDER BY rowid DESC LIMIT ? OFFSET ?", 
                (limit, offset)
            )
            for row in cursor:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "tags": row[2].split() if row[2] else [],
                    "note": row[3]
                })
        return results
    except Exception as e:
        logging.error(f"List memories error: {e}")
        return []

@app.tool("delete_memory")
def delete_memory(memory_id: str) -> str:
    """根据ID永久删除一条记忆"""
    logging.info(f"Tool called: delete_memory | ID: {memory_id}")
    try:
        # 1. 删除 SQLite 记录
        if conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            logging.info("Deleted from SQLite")

        # 2. 删除 LanceDB 记录
        if vector_table:
            try:
                # LanceDB 的删除语法：delete("SQL-like filter")
                vector_table.delete(f"id = '{memory_id}'")
                logging.info("Deleted from LanceDB")
            except Exception as le:
                logging.warning(f"LanceDB delete warning (might not exist): {le}")

        return f"Memory {memory_id} deleted successfully."
    except Exception as e:
        logging.error(f"Delete error: {e}")
        return f"Error deleting memory: {e}"

if __name__ == "__main__":
    # 使用 SSE 模式启动
    # host="0.0.0.0" 允许外部连接，port=8000
    logging.info("Starting SSE Server on port 8000...")
    app.run(transport="sse", host="0.0.0.0", port=8000)
