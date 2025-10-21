# Graph Tokenization Framework

This repository provides tools for **synthetic graph generation** and **task-specific graph tokenization** across multiple network types and reasoning tasks. It converts graphs into **text-based token sequences** suitable for training and evaluating large language models (LLMs)/ Pure Transformers on graph reasoning.

---

## Overview

We generate graphs from a variety of well-known random graph models, then tokenize them into sequences representing their structure and properties.  
Each task defines its own reasoning format (e.g., node degree prediction, reachability, edge existence).

---

## Supported Graph Types

| Algorithm | Description |
|------------|-------------|
| **er** | *Erdős–Rényi random graphs* — edges are formed independently with uniform probability. |
| **ba** | *Barabási–Albert scale-free graphs* — generated via preferential attachment, forming hub nodes. |
| **sbm** | *Stochastic Block Model* — community-structured graphs with dense intra-cluster and sparse inter-cluster edges. |
| **sfn** | *Scale-Free Network (Holme–Kim / Power-Law)* — mimics real-world social and biological networks with high-degree hubs. |
| **complete** | *Complete graphs* — every node is connected to every other node. |
| **star** | *Star graphs* — a single central hub connected to all leaf nodes. |
| **path** | *Path (chain) graphs* — nodes arranged in a linear sequence. |

---

## Supported Graph Reasoning Tasks

Each graph task defines a unique reasoning objective applied to the generated graphs.  
These tasks are implemented in `graph_task.py` and can be selected via the `--task` argument when running the task generator.

### Available Tasks

| Task | Description |
|------|-------------|
| **EdgeExistence** | Determines whether an edge exists between two given nodes (`<q> u v <p> yes/no`). |
| **NodeDegree** | Predicts the degree (number of edges) of a given node (`<q> node_id <p> d[k]`). |
| **NodeCount** | Counts the total number of nodes in the graph (`<q> node_count <p> n[k]`). |
| **EdgeCount** | Counts the total number of edges in the graph (`<q> edge_count <p> m[k]`). |
| **ConnectedNodes** | Lists all nodes connected to a specific node (`<q> neighbors u <p> {v1 v2 ...}`). |
| **CycleCheck** | Checks whether the graph contains a cycle (`<q> has_cycle <p> yes/no`). |
| **DisconnectedNodes** | Identifies isolated or disconnected nodes in the graph (`<q> isolated_nodes <p> {…}`). |
| **Reachability** | Determines whether a path exists between two nodes (`<q> u v <p> yes/no`). |
| **ShortestPath** | Computes the shortest path length between two nodes (`<q> u v <p> len[k]`). |
| **MaximumFlow** | Calculates the maximum flow between two nodes using unit capacities (`<q> u v <p> f[k]`). |
| **TriangleCounting** | Counts the total number of triangles in the graph (`<q> triangle_count <p> t[k]`). |
| **NodeClassification** | Predicts the community or class label of each node (e.g., from an SBM graph) (`<q> class u <p> c[k]`). |

---

## Graph Tokenization Process

Each graph is represented as a **token sequence** capturing its structure and the reasoning task.

**Example format:**
```text
<bos> 0 1 <e> 1 2 <e> ... <n> 0 1 2 ... <q> [source] <p> [prediction] <eos>
```

**Where:**
- `<bos>` → Begin of sequence  
- `i j <e>` → Edge between nodes *i* and *j*  
- `<n>` → Start of node list  
- `<q>` → Query node(s) or task input  
- `<p>` → Prediction or label output  
- `<eos>` → End of sequence  

### Examples
| Task | Tokenized Example |
|------|-------------------|
| **Node Degree** | `<bos> 0 1 <e> 1 2 <e> <n> 0 1 2 <q> 0 <p> d2 <eos>` |
| **Reachability** | `<bos> 0 1 <e> 1 2 <e> <n> 0 1 2 <q> 0 2 <p> yes <eos>` |
| **Edge Existence** | `<bos> 0 1 <e> 1 2 <e> <n> 0 1 2 <q> 0 3 <p> no <eos>` |

---

## Running the Code

### 1. Generating Graphs

Generate random graphs for each algorithm and dataset split (`train`, `valid`, `test`):

```sh
./graphtoken/graph_generator.sh
```


The generated files are stored in: `graphs/<algorithm>/<split>/*.graphml`

### 2. Generating Task Files

Tokenize the generated graphs into JSON files for specific graph reasoning tasks:

```sh
./graphtoken/task_generator.sh
```

This will create task-specific tokenized samples in: `tasks/<task>/<algorithm>/<split>/<graph_id>.json`

Each `.json` file contains a list of tokenized samples, for example:

```json
[
  {
    "graph_id": "er_test_0",
    "text": "<bos> 0 1 <e> 1 2 <e> ... <n> 0 1 2 <q> 0 <p> d2 <eos>"
  },
  {
    "graph_id": "er_test_0",
    "text": "<bos> 0 1 <e> 1 2 <e> ... <n> 0 1 2 <q> 1 <p> d1 <eos>"
  }
]
```

### Output Structure

After running the task generator, the tokenized graph files will be saved in the following directory format:

graphtoken/  
│  
├── `graph_generator.sh` # Shell script for graph generation  
├── `task_generator.sh` # Shell script for task-based tokenization  
│  
├── `graph_generator.py` # Generates and saves random graphs  
├── `graph_task_generator.py` # Tokenizes graphs for each reasoning task  
├── `graph_task.py` # Defines all task classes (NodeDegree, Reachability, etc.)  
├── `graph_task_utils.py` # I/O utilities (pure Python, no TensorFlow dependency)  
│  
├── `graphs/` # Generated raw graphs (`.graphml`) → `graphs/<algorithm>/<split>/`  
│  
└── `tasks/` # Tokenized task datasets (`.json`) → `tasks/<task>/<algorithm>/<split>/`



## Contact us

For questions or comments about the implementation, please contact alparviz@ucsd.edu.