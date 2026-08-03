# Multi-Intent Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow one user message to produce an ordered list of intents that update dependent context, reuse existing retrieval/state nodes, and return one merged answer.

**Architecture:** Replace the single-turn intent result with an LLM-produced, Pydantic-validated action plan. Execute its actions in dependency order, snapshot each action result before the next action overwrites shared retrieval fields, and generate one final answer from the accumulated results. Preserve the current `CookingState` rules: recommendation does not alter cooking progress.

**Tech Stack:** Python, Pydantic, LangGraph, SQLite checkpoint, Qdrant hybrid retrieval.

## Global Constraints

- Keep `recommendation` as an existing action type; do not introduce a separate recommendation subgraph.
- Use LLM structured output and Pydantic validation for action identification and ordering; do not classify by keywords.
- Each retrieval action may trigger at most one evidence retry.
- Persist preference updates before dependent recommendation/query rewriting in the same turn.
- Keep one user-facing final answer and preserve citations per action.

---

### Task 1: Define validated action-plan state

**Files:**
- Modify: `src/dish_rag/models.py`
- Modify: `src/dish_rag/agent/state.py`
- Test: `tests/test_multi_intent.py`

**Interfaces:**
- Produces `IntentAction`, `IntentPlan`, and `ActionResult` Pydantic models.
- Produces state keys `intent_plan`, `current_action_index`, `action_results`, and action-scoped retry state.

- [ ] **Step 1: Write failing tests**

```python
def test_action_plan_preserves_dependency_order():
    plan = IntentPlan.model_validate({
        "actions": [
            {"intent": "preference_update"},
            {"intent": "recommendation", "recommendation_count": 3},
        ]
    })
    assert [action.intent for action in plan.actions] == [
        Intent.PREFERENCE_UPDATE,
        Intent.RECOMMENDATION,
    ]
```

- [ ] **Step 2: Run the test and confirm it fails because the action-plan models do not exist.**
- [ ] **Step 3: Add the models and state keys with defaults that preserve old single-intent checkpoints.**
- [ ] **Step 4: Run the focused test and confirm it passes.**

### Task 2: Classify and execute actions sequentially

**Files:**
- Modify: `src/dish_rag/agent/prompts.py`
- Modify: `src/dish_rag/agent/nodes.py`
- Modify: `src/dish_rag/agent/graph.py`
- Test: `tests/test_multi_intent.py`

**Interfaces:**
- Consumes `IntentPlan.actions[current_action_index]`.
- Produces one `ActionResult` per completed action and increments `current_action_index`.

- [ ] **Step 1: Write failing tests**

```python
def test_preference_action_updates_memory_before_recommendation_rewrite():
    result = run_actions("我健身、不吃花生，推荐3道高蛋白菜")
    assert result.action_results[0].intent == Intent.PREFERENCE_UPDATE
    assert result.action_results[1].query_rewrite.preserved_constraints == ["不吃花生"]
```

- [ ] **Step 2: Run the test and confirm it fails because the graph has no action executor.**
- [ ] **Step 3: Make the classifier output an ordered action list, execute one action at a time, and refresh in-turn memory after a preference action.**
- [ ] **Step 4: Route each retrieval action through the existing rewrite, retrieve, rerank, evidence-judge, and one-retry logic. Reset retry state for the next action.**
- [ ] **Step 5: Run the focused test and confirm it passes.**

### Task 3: Preserve state, HITL, and action-specific results

**Files:**
- Modify: `src/dish_rag/agent/nodes.py`
- Modify: `src/dish_rag/agent/graph.py`
- Test: `tests/test_multi_intent.py`

**Interfaces:**
- `recommendation` returns distinct recipe evidence without changing `CookingState`.
- State-mutating actions retain the existing cooking-state behavior.

- [ ] **Step 1: Write failing tests**

```python
def test_navigation_then_recommendation_keeps_navigation_cooking_state():
    result = run_actions("下一步，并推荐3道高蛋白菜", active_recipe="001", step_no=2)
    assert result.cooking_state.current_step_no == 3
    assert result.action_results[1].intent == Intent.RECOMMENDATION
```

- [ ] **Step 2: Run the test and confirm it fails because later action fields overwrite prior results.**
- [ ] **Step 3: Snapshot each action result, allow only existing stateful intents to modify `CookingState`, and resume the same action after HITL.**
- [ ] **Step 4: Run the focused test and confirm it passes.**

### Task 4: Aggregate answer, trace, documentation, and regression coverage

**Files:**
- Modify: `src/dish_rag/agent/prompts.py`
- Modify: `src/dish_rag/agent/nodes.py`
- Modify: `src/dish_rag/observability.py`
- Modify: `README.md`
- Test: `tests/test_multi_intent.py`

**Interfaces:**
- Consumes `action_results`.
- Produces one answer ordered by actions and grouped citations.

- [ ] **Step 1: Write failing tests**

```python
def test_final_answer_is_generated_once_from_all_action_results():
    result = run_actions("记住我不吃花生，并推荐2道菜")
    assert result.answer_calls == 1
    assert len(result.citations) == 2
```

- [ ] **Step 2: Run the test and confirm it fails because the answer node only sees the final action.**
- [ ] **Step 3: Render action results into one answer prompt, record per-action trace data, and document ordering, state, retry, and HITL semantics.**
- [ ] **Step 4: Run focused and full tests, then run syntax and whitespace checks.**
