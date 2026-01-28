# Implementation Plan: Search Intelligence & Hybrid Optimization

## Phase 1: Search Engine Architecture
- [x] Task: Create `search_engine.py` and migrate `embed` logic
    - [x] Move resource initialization (Session, Table) if appropriate, or pass them as dependencies.
    - [x] Write unit test for `SearchService.embed`.
- [x] Task: Implement Similarity Scoring
    - [x] Add logic to convert distance to 0-1 similarity score.
    - [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Hybrid Fusion Implementation
- [x] Task: Implement RRF algorithm
    - [x] Create `ranking.py` or similar for pure math functions.
    - [x] Write test cases for RRF with predictable inputs.
- [x] Task: Integrate RRF into `SearchService`
    - [x] Fetch multiple sources, apply RRF, sort, and slice.
- [x] Task: Implement Threshold Filtering
    - [x] Apply threshold to vector results before fusion.
- [x] Task: Update `server.py` to use `SearchService`
    - [x] Replace inline logic in `search_memory` tool.
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)
