from fastmcp import FastMCP
from typing import List, Dict
import uuid

app = FastMCP("Project Memory Bank (Prototype)")

# 内存存储：list of dicts
memories: List[Dict] = []

@app.tool("save_memory")
def save_memory(content: str, tags: List[str] = None, note: str = "") -> str:
    """保存一条项目记忆（如bug总结、架构决策、代码片段）"""
    memory = {
        "id": str(uuid.uuid4()),
        "content": content,
        "tags": tags or [],
        "note": note
    }
    memories.append(memory)
    return f"Memory saved with id: {memory['id']}"

@app.tool("search_memory")
def search_memory(query: str, top_k: int = 5) -> List[Dict]:
    """基于简单关键词匹配搜索记忆"""
    query_lower = query.lower()
    results = []
    for mem in memories:
        if (query_lower in mem["content"].lower() or
            any(query_lower in tag.lower() for tag in mem["tags"])):
            results.append({
                "id": mem["id"],
                "content": mem["content"],
                "note": mem["note"],
                "tags": mem["tags"]
            })
        if len(results) >= top_k:
            break
    return results

# stdio模式启动
if __name__ == "__main__":
    app.run()