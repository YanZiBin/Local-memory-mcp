# Initial Concept
开发一个本地化、隐私安全的长效记忆 MCP 服务器，采用双库混合检索。

# Product Guide - Local Memory MCP Server

## Vision
为 AI Agent（如 Gemini CLI, Claude Desktop 以及未来可能的 VS Code/Cursor 集成）提供一个完全本地化、隐私安全、永久存储且具备无限容量的长效记忆系统。通过 Model Context Protocol (MCP) 与 Server-First（手动启动）模式，实现数据主权与高性能 AI 辅助的完美结合。

## Core Goals
1. **数据主权**：所有记忆数据严格存储在用户本地，不经过云端，零隐私风险。
2. **混合检索**：通过 LanceDB (向量) 与 SQLite FTS5 (全文) 的结合，提供高精度的语义和关键词检索。
3. **高性能**：充分利用本地 GPU (TensorRT/CUDA) 加速嵌入生成，确保毫秒级响应。
4. **稳定性优先**：采用手动启动 Server 的模式，避免由客户端自动拉起进程可能导致的环境冲突或超时，确保重型模型加载的可靠性。

## Target Users
- 对隐私极其敏感的开发者和 AI 进阶用户。
- 希望在不同项目间共享长效知识库的软件工程师。
- 需要在无网络或离线环境下使用高性能 AI 辅助的专业人士。

## Key Features (Phased)
- **记忆存取**：通过 `save_memory` 和 `search_memory` 工具实现片段、代码、事实的持久化与检索。
- **智能筛选 (规划中)**：引入 RRF 融合算法、相似度阈值以及自动权重提升（根据检索频率）。
- **架构守栏 (规划中)**：通过 ADR 召唤实现“温和提醒”式的开发规范守卫。
- **管理工具**：提供 `list_memories` 和 `delete_memory` 进行日常维护。

## Future Integration
- 深度集成主流 IDE（如 VS Code, Cursor）。
