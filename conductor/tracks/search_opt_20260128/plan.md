# Implementation Plan: Search Intelligence & Hybrid Optimization

## Phase 1: Search Engine Architecture
- [ ] Task: Create `search_engine.py` and migrate `embed` logic
    - [ ] Move resource initialization (Session, Table) if appropriate, or pass them as dependencies.
    - [ ] Write unit test for `SearchService.embed`.
- [ ] Task: Implement Similarity Scoring
    - [ ] Add logic to convert distance to 0-1 similarity score.
    - [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Hybrid Fusion Implementation
- [ ] Task: Implement RRF algorithm
    - [ ] Create `ranking.py` or similar for pure math functions.
    - [ ] Write test cases for RRF with predictable inputs.
- [ ] Task: Integrate RRF into `SearchService`
    - [ ] Fetch multiple sources, apply RRF, sort, and slice.
- [ ] Task: Implement Threshold Filtering
    - [ ] Apply threshold to vector results before fusion.
- [ ] Task: Update `server.py` to use `SearchService`
    - [ ] Replace inline logic in `search_memory` tool.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)
