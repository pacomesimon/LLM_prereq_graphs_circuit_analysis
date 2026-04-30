---
title: Performance-Aware Model Compression for Circuit Analysis
emoji: 🕸️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: mit
short_description: Strategic LLM evaluation for circuit analysis via prerequisite graphs.
---

# 🕸️ Performance-Aware Model Compression for Circuit Analysis Using Prerequisite Graphs

This is a strategic LLM evaluation framework designed to test **model cascades** on hierarchical task graphs. Tasks are modeled as **Directed Acyclic Graphs (DAGs)**, representing complex real-world workflows where success on a parent node is a prerequisite for attempting its descendants.

---

## 🏗️ Technical Reproduction Guide

This section provides the implementation specifications required to replicate the framework's core features.

### 1. Agentic Dataset Generation
**Phase 1: The Blueprint (Conceptual Inheritance)**
-   **Logic**: Use an LLM agent to generate a tree of "concept nodes."
-   **Inheritance**: Every child node MUST inherit the `tags` of its parent and append 1-2 unique "complexity tags" (e.g., if parent is `Auth`, child is `Auth + Rate-Limiting`).
-   **Schema**: Nodes must contain `id`, `parent_id`, `tags`, and a `description`.

**Phase 2: Q&A Translation (Tool-Calling MCQs)**
-   **Logic**: A separate agent iterates over each blueprint node.
-   **Transformation**: It converts the node's description and tags into a specific MCQ with 4 choices.
-   **Ground Truth**: The agent must provide the `correct_idx` (0-3) as a structured tool call response.

### 2. Strategic Evaluation Engine (The Cascade)
**DFS State Machine**
-   **Initialization**: Sort LLMs by capability (e.g., `[nano, mini, pro]`).
-   **Traversal**: Use Depth-First Search (DFS) starting from all root nodes (`parent_id: null`).
-   **Upgrade Trigger**: 
    -   Track `path_failures` (cumulative failures from root to current node).
    -   If `path_failures > Threshold`, increment the `model_index` (upgrade to a larger model).
-   **State Persistence**: Once a branch is upgraded, all subsequent children on that branch use the new model (or better) to ensure consistency.
-   **Pruning**: If the largest model in the list fails a node, mark it as `failed_all` and skip all its descendants (`skipped`).

### 3. Visual Analysis Math
**Radial Graph Visualization**
-   **Algorithm**: Group nodes by their BFS distance from roots.
-   **Layout**: Map these groups to concentric shells in a `nx.shell_layout`.
-   **Styling**: Use colors to represent node status (Success: Blue, Fail/Skip: Red).

**Tag-Set Venn Diagrams**
-   **Categorization**: Group every unique tag into sets based on outcome: `Model_N (Pass)`, `Model_N (Fail)`, `Failed All`, `Skipped`.
-   **Intersections**: Calculate set overlaps (Intersection/Difference) to identify specific conceptual boundaries of model intelligence.

### 4. The Monotonic Capability Assumption
This optional analytical mode applies the following formal assumptions to the results:
-   **Success Inheritance**: $Pass_{Larger} = Pass_{Larger} \cup Pass_{Smaller}$. If a small model solves it, we assume the larger model would also solve it.
-   **Nested Failure**: $FailedAll = Failures_{LargestModel}$. We assume if the largest model failed, it is a global failure.
-   **Complexity Horizon**: $Skipped = Skipped_{Actual} \cup FailedAll$. We assume that if the best model failed a node, all its descendants are beyond the system's current reach.

---

## 📦 Setup & Installation
1.  **Requirements**:
    ```bash
    pip install gradio networkx langchain langchain-openai langchain-ollama matplotlib matplotlib-venn pandas python-dotenv
    ```
2.  **Environment**: Add your keys to `.env`:
    - `OPENAI_API_KEY=...`
3.  **Local LLMs**: Ensure **Ollama** is running if using local models.
4.  **Run**:
    ```bash
    python app.py
    ```
