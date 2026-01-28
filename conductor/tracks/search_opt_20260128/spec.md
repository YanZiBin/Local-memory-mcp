# Specification: Search Intelligence & Hybrid Optimization

## Context
The current `search_memory` tool performs independent searches in SQLite and LanceDB but merges them without sophisticated ranking. This leads to sub-optimal recall and potential noise.

## Goals
1.  **RRF Ranking**: Use Reciprocal Rank Fusion to combine results from multiple sources fairly.
2.  **Noise Reduction**: Filter low-confidence vector matches.
3.  **Refactoring**: Improve codebase maintainability by decoupling search logic from the MCP app definition.

## Requirements

### 1. Search Service Refactoring
-   Move `embed` function and search logic to a dedicated service.
-   Interface: `SearchService.hybrid_search(query, top_k, threshold, rrf_k)`.

### 2. Reciprocal Rank Fusion (RRF)
-   Formula: `Score(d) = sum(1 / (k + rank(d, source)))` for each source.
-   Default `k` = 60.
-   Top-K results (e.g., 20) should be fetched from each source before fusion.

### 3. Similarity Thresholding
-   LanceDB returns distances. Convert to similarity score (e.g., `1 - distance` for normalized vectors).
-   Filter results where `similarity < threshold`. Default threshold: 0.7.

## Non-Functional Requirements
-   Keep response time < 300ms.
-   Ensure no change to tool input/output schema (Backward compatibility).
