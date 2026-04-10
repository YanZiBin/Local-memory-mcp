# Local Memory MCP Server (Project Memory Bank)

[English](#english) | [中文说明](#chinese)

---

<a name="english"></a>
## 🇬🇧 English Description

### 🌟 Design Philosophy & Motivation

**"Your memory belongs to you, not the cloud."**

I built this project with a simple yet powerful goal: **Total Data Sovereignty**.
In an era of subscription-based AI services and cloud dependencies, I wanted a solution that is:

1.  **100% Local & Private:** No data ever leaves your machine. No API fees, no privacy risks.
2.  **Permanent:** As long as your hard drive exists, your AI's memory exists. No fear of service shutdowns.
3.  **Infinite Capacity:** The only limit is your local disk space.
4.  **High Performance:** Utilizing local GPU acceleration (TensorRT/CUDA) for lightning-fast embedding and retrieval.

This is a **Memory Context Protocol (MCP)** server that gives your AI (like Gemini CLI, Claude Desktop) a persistent, searchable, and evolving long-term memory.

### ✨ Key Features

*   **Hybrid Search Architecture:** Combines **LanceDB** (Vector Search for semantic understanding) and **SQLite FTS5** (Full-Text Search for exact keyword matching) for high-precision recall.
*   **Hardware Acceleration:** Powered by ONNX Runtime with TensorRT/CUDA execution providers for millisecond-level embedding generation.
*   **Standard MCP Tools:**
    *   `save_memory`: Store snippets, code, docs, or personal facts (with automatic duplicate detection).
    *   `search_memory`: Semantic & keyword retrieval.
    *   `list_memories`: View recent entries.
    *   `delete_memory`: Manage and clean up data.
    *   `update_memory`: Update existing memory by ID.
*   **Lazy Loading:** Optimized startup time with on-demand resource initialization.
*   **Zero Cost:** Runs entirely on your existing hardware.

### 🛠️ Prerequisites

*   **OS:** Windows (tested)
*   **Python:** 3.10 or higher.
*   **Hardware:** NVIDIA GPU recommended (for TensorRT/CUDA acceleration), but works on CPU.
*   **MCP Client:** [Gemini CLI](https://github.com/google-gemini/gemini-cli) or [Claude Desktop](https://claude.ai/download).Or any IDE that can be configured with MCP.

### 🚀 Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/YanZiBin/Local-memory-mcp.git
cd Local-memory-mcp
```

#### 2. Create a Python Environment (Conda Recommended)
To ensure GPU libraries work correctly, Conda is highly recommended.
```bash
conda create -n Local-memory-mcp python=3.10
conda activate Local-memory-mcp
```

#### 3. Install Dependencies
```bash
pip install fastmcp lancedb onnxruntime-gpu transformers numpy uvicorn
```
*(Note: If you don't have a GPU, install `onnxruntime` instead of `onnxruntime-gpu`)*

#### 4. Download the Embedding Model
This project uses `BAAI/bge-m3` converted to ONNX[here](https://huggingface.co/Xenova/bge-m3/tree/main). You need to download the model files into the `bge-m3-onnx` directory.

You can use `huggingface-cli` or manually download these files:
*   `config.json`
*   `model.onnx`
*   `model.onnx_data`
*   `vocab.txt`
*   `sentence_transformers.onnx`
*   `sentence_transformers.onnx_data`
*   `tokenizer_config.json`
*   `tokenizer.json`

Place them inside a folder named `bge-m3-onnx` in the project root.

### 🏃‍♂️ Running the Server

Since this server uses heavy local models, we recommend the **Manual Start (SSE Mode)** for stability.

1.  **Start the Server:**
    Open a terminal and run:
    ```bash
    python server.py
    ```
    Wait until you see: `INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

2.  **Connect your Client (e.g., Gemini CLI):**
    
    Edit your Gemini CLI configuration file (usually at `~/.gemini/settings.json` or `%USERPROFILE%\.gemini\settings.json` on Windows):

    ```json
    {
      "mcpServers": {
        "local-memory": {
          "url": "http://localhost:8000/sse",
          "type": "sse"
        }
      }
    }
    ```

3.  **Start using it!**
    Open Gemini CLI and try:
    > "Save this memory: My project uses Python 3.10."
    > "Search my memories for 'project'."

### 🗺️ Roadmap

We are currently transitioning to **Phase 4 (Memory Management)**.

- [x] **Phase 1: Prototype**
    - Initialize MCP server with `fastmcp`.
    - Basic in-memory storage (dict).
    - Implement basic `save_memory` & `search_memory` (keyword matching).
    - Manual connection testing with Gemini CLI.

- [x] **Phase 2: Persistence**
    - Integrate **SQLite FTS5** and **LanceDB**.
    - Integrate local **ONNX embedding model**.
    - Dual-storage architecture (Full-text + Vector).
    - `list_memories` and `delete_memory` tools.

- [x] **Phase 3: Intelligent Retrieval (Completed)**
    - [x] Implement **RRF (Reciprocal Rank Fusion)** algorithm for high-quality hybrid search.
    - [x] Add similarity thresholds & Top-K limits to reduce noise.
    - [ ] Implement **Contextual Retrieval** (Enhanced context storage).
    - [ ] (Optional) Add Reranker for higher precision.

- [ ] **Phase 4: Memory Management**
    - [x] Duplicate detection on save (similarity-based).
    - [ ] Lifecycle management (Time decay, Conflict tagging).

- [ ] **Phase 5: Advanced Optimization**
    - [ ] Expose **Resources**: Project summaries, ADR (Architecture Decision Records) guardrails.
    - [ ] Architectural guardrails (Recall ADRs on violation) & Task chain tracking.

---

<a name="chinese"></a>
## 🇨🇳 中文说明

### 🌟 设计初衷

**“你的记忆属于你，而不是云端。”**

开发这个项目的初衷非常纯粹：实现**完全的数据主权**。
在这个万物订阅制、隐私担忧日益严重的时代，我希望构建一个这样的解决方案：

1.  **完全本地化 & 隐私安全：** 没有任何数据会上传云端。没有 API 调用费，没有隐私泄露风险。
2.  **永久存储：** 只要你的硬盘还在，你的 AI 记忆就在。不必担心服务商倒闭或“跑路”。
3.  **无限容量：** 唯一的限制是你本地硬盘的大小（相当于无限）。
4.  **极致性能：** 利用本地 GPU 加速（TensorRT/CUDA），实现毫秒级的记忆存取。

这是一个 **MCP (Model Context Protocol)** 服务器，它为你的 AI 工具（如 Gemini CLI, Claude Desktop）提供了一个持久化、可搜索、不断进化的“外脑”。

### ✨ 核心功能

*   **混合搜索架构：** 结合了 **LanceDB**（向量搜索，理解语义）和 **SQLite FTS5**（全文搜索，精准匹配关键词），并通过 **RRF (倒数排名融合)** 算法进行智能排序，确保召回率和准确率。
*   **硬件加速：** 基于 ONNX Runtime 和 TensorRT/CUDA，充分释放本地显卡性能。
*   **标准 MCP 工具集：**
    *   `save_memory`: 保存代码片段、文档总结或个人事实（自动检测重复内容）。
    *   `search_memory`: 语义或关键词检索（支持相似度阈值过滤）。
    *   `list_memories`: 查看最近的记忆。
    *   `delete_memory`: 删除过时信息。
    *   `update_memory`: 按 ID 更新现有记忆。
*   **懒加载设计 (Lazy Loading)：** 优化启动流程，按需加载重型模型，拒绝卡顿。
*   **零成本：** 以前需要付费购买的向量存储服务，现在免费运行在你自己的电脑上。

### 🛠️ 环境要求

*   **操作系统：** Windows (已充分测试)
*   **Python：** 3.10 或更高版本。
*   **硬件：** 推荐使用 NVIDIA 显卡（以获得 TensorRT/CUDA 加速），但也支持 CPU 运行。
*   **MCP 客户端：** [Gemini CLI](https://github.com/google-gemini/gemini-cli) 或 [Claude Desktop](https://claude.ai/download)。或者任何可以配置mcp的IDE。

### 🚀 安装与配置指南

#### 1. 克隆项目
```bash
git clone https://github.com/YanZiBin/Local-memory-mcp.git
cd Local-memory-mcp
```

#### 2. 创建 Python 环境 (强烈推荐 Conda)
为了避免 CUDA 依赖冲突，建议使用 Conda。
```bash
conda create -n Local-memory-mcp python=3.10
conda activate Local-memory-mcp
```

#### 3. 安装依赖库
```bash
pip install fastmcp lancedb onnxruntime-gpu transformers numpy uvicorn
```
*（注：如果你没有 NVIDIA 显卡，请将 `onnxruntime-gpu` 替换为 `onnxruntime`）*

#### 4. 下载嵌入模型 (Embedding Model)
本项目使用 `BAAI/bge-m3` 的 ONNX 量化版本[here](https://huggingface.co/Xenova/bge-m3/tree/main)。你需要将模型文件下载到项目根目录下的 `bge-m3-onnx` 文件夹中。

你可以使用 `huggingface-cli` 或手动下载以下文件：
*   `config.json`
*   `model.onnx`
*   `model.onnx_data`
*   `vocab.txt`
*   `sentence_transformers.onnx`
*   `sentence_transformers.onnx_data`
*   `tokenizer_config.json`
*   `tokenizer.json`

确保它们都在 `bge-m3-onnx` 文件夹内。

### 🏃‍♂️ 运行与使用

由于本项目加载了本地大模型，为了稳定性，我们推荐使用 **手动启动 (SSE 模式)**。

1.  **启动服务器：**
    打开终端（CMD/PowerShell），运行：
    ```bash
    python server.py
    ```
    等待直到看到提示：`INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

2.  **连接客户端 (以 Gemini CLI 为例)：**
    
    编辑你的 Gemini CLI 配置文件（通常位于 `~/.gemini/settings.json` 或 Windows 的 `%USERPROFILE%\.gemini\settings.json`）：

    ```json
    {
      "mcpServers": {
        "local-memory": {
          "url": "http://localhost:8000/sse",
          "type": "sse"
        }
      }
    }
    ```

3.  **开始体验！**
    打开 Gemini CLI，直接对话：
    > “帮我记住：我的项目运行在 Python 3.10 环境下。”
    > “搜索记忆：关于项目环境的信息。”

### 🗺️ 开发路线图 (Roadmap)

目前项目正过渡到 **第四阶段：记忆管理**。

- [x] **第一阶段：原型验证**
    - 使用 `fastmcp` 初始化 MCP 服务器。
    - 使用内存字典进行临时存储。
    - 实现基础的 `save_memory` 和 `search_memory`（关键词匹配）。
    - 配置 stdio 并在 Gemini CLI 中测试手动连接。

- [x] **第二阶段：持久化存储**
    - 引入 **SQLite FTS5** 和 **LanceDB**。
    - 集成本地 **ONNX 嵌入模型** 生成向量。
    - 实现双库存储（全文 + 向量）。
    - 新增 `list_memories`（列出记忆）和 `delete_memory`（删除记忆）工具。

- [x] **第三阶段：智能筛选 (已完成)**
    - [x] 实现 **RRF (倒数排名融合)** 算法，实现高质量混合检索。
    - [x] 增加相似度阈值和 Top-K 限制，减少噪声。
    - [ ] 实现 **Contextual Retrieval**（增强文本存储）。
    - [ ] （可选）加入 Reranker 重排序模型以提升精度。

- [ ] **第四阶段：记忆管理**
    - [x] 保存时自动检测重复内容（基于相似度）
    - [ ] 实现生命周期管理（时间衰减、冲突标记）。

- [ ] **第五阶段：高级优化**
    - [ ] 暴露 **Resources**（资源）：项目摘要、ADR 架构决策记录守栏等。
    - [ ] 实现架构守栏（违规时自动召回 ADR）和任务链跟踪。

---

**License:** MIT
**Author:** [YanZiBin]
