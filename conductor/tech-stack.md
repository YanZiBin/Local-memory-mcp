# Tech Stack - Local Memory MCP Server

## Core Technologies
- **Programming Language**: Python 3.10+
- **MCP Framework**: FastMCP
- **Vector Database**: LanceDB (with pyarrow support)
- **Full-Text Search**: SQLite FTS5
- **Inference Engine**: ONNX Runtime
- **Hardware Acceleration**: TensorRT / CUDA Execution Providers

## Key Dependencies
- `lancedb`: Vector storage and retrieval.
- `pyarrow`: High-performance data interchange format.
- `onnxruntime-gpu`: Accelerated embedding generation.
- `fastmcp`: Standard interface for Model Context Protocol.
- `pandas`: Data manipulation and integration with LanceDB.
