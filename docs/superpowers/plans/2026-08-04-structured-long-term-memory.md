# Structured Long-Term Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep at most ten semantically distinct recent preferences per user while retaining restrictions until explicitly removed.

**Architecture:** Replace the single overwrite-only `preference_latest` payload with structured preference and restriction records in SQLite. A dedicated, Pydantic-validated LLM memory operation resolves semantic merge/add/remove decisions; generic Agent prompts receive only the active preference phrases and restrictions.

**Tech Stack:** Python, Pydantic, SQLite, LangGraph, OpenAI-compatible ChatClient, pytest.

## Global Constraints

- Preference records retain their original phrases and refresh `last_seen_at` on a semantic merge.
- Only ten distinct active preferences are retained per user; restrictions are never pruned by count.
- A restriction may be removed only by an explicit removal operation from the memory-resolution model.
- Raw query and completed query remain audit data and are never injected into generic LLM prompts.

### Task 1: Add structured persistence and models

**Files:** `src/dish_rag/models.py`, `src/dish_rag/storage/sqlite_store.py`, `tests/test_long_term_memory.py`

- [ ] Write tests for adding, merging, pruning preferences and adding/removing restrictions.
- [ ] Add Pydantic memory item, snapshot, and operation models.
- [ ] Add SQLite migration and CRUD methods for `user_preferences` and `user_restrictions`.
- [ ] Run the focused tests.

### Task 2: Resolve and apply memory operations

**Files:** `src/dish_rag/agent/prompts.py`, `src/dish_rag/agent/nodes.py`, `tests/test_long_term_memory.py`

- [ ] Write tests proving semantically equivalent phrases merge, explicit removal deletes only the matching restriction, and a new preference is capped at ten.
- [ ] Add a structured memory-resolution prompt and validate its JSON response.
- [ ] Route both legacy and multi-intent `preference_update` paths through one memory-update method.
- [ ] Run the focused tests.

### Task 3: Prompt boundary, migration, and documentation

**Files:** `src/dish_rag/agent/nodes.py`, `README.md`, `tests/test_long_term_memory.py`

- [ ] Write tests proving generic prompts receive only active preference/restriction phrases, not audit fields.
- [ ] Convert existing `preference_latest` payloads once on read, conservatively retaining unclassified legacy constraints as restrictions.
- [ ] Update README with retention, semantic merge, explicit removal, and migration rules.
- [ ] Run the full test suite and static checks.
