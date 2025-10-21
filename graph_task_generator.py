# coding=utf-8
# Copyright 2025 ...
# Licensed under the Apache License, Version 2.0

r"""Generate task-specific tokenized graph samples (simple JSON output)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json

from absl import app
from absl import flags
from absl import logging
import networkx as nx
import numpy as np

from graph_task import EdgeExistence, NodeDegree, NodeCount, EdgeCount
from graph_task import ConnectedNodes, CycleCheck, DisconnectedNodes
from graph_task import Reachability, ShortestPath, MaximumFlow
from graph_task import TriangleCounting, NodeClassification 
import graph_task_utils as utils


_TASK = flags.DEFINE_enum(
    "task",
    None,
    [
        "edge_existence",
        "node_degree",
        "node_count",
        "edge_count",
        "connected_nodes",
        "cycle_check",
        "disconnected_nodes",
        "reachability",
        "shortest_path",
        "maximum_flow",
        "triangle_counting",
        "node_classification",
    ],
    "The task to generate datapoints.",
    required=True,
)
_ALGORITHM = flags.DEFINE_enum(
    "algorithm",
    None,
    ["er", "ba", "sbm", "sfn", "complete", "star", "path", "all"],
    "The graph generator algorithm(s) to read from.",
    required=True,
)
_TASK_DIR = flags.DEFINE_string("task_dir", None, "Root directory to write tasks.", required=True)
_GRAPHS_DIR = flags.DEFINE_string("graphs_dir", None, "Root directory containing graphs.", required=True)
_SPLIT = flags.DEFINE_enum("split", "test", ["train", "valid", "test"], "Which split to read from.")
_RANDOM_SEED = flags.DEFINE_integer("random_seed", 1234, "Random seed for any sampling.")


TASK_CLASS = {
    'edge_existence': EdgeExistence,
    'node_degree': NodeDegree,
    'node_count': NodeCount,
    'edge_count': EdgeCount,
    'connected_nodes': ConnectedNodes,
    'cycle_check': CycleCheck,
    'disconnected_nodes': DisconnectedNodes,
    'reachability': Reachability,
    'shortest_path': ShortestPath,
    'maximum_flow': MaximumFlow,
    'triangle_counting': TriangleCounting,
    'node_classification': NodeClassification,
}

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _graph_id_from_index(algorithm: str, split: str, idx: int) -> str:
    return f"{algorithm}_{split}_{idx}"


def _regenerate_sbm_graphs_like(graphs, rng: np.random.RandomState):
    """Regenerate SBM graphs of similar size (for NodeClassification)."""
    regenerated = []
    for g in graphs:
        n = g.number_of_nodes()
        if n < 4:
            n = 6
        sizes = [n // 2, n - n // 2]
        p_in = rng.uniform(0.6, 0.8)
        p_out = rng.uniform(0.0, 0.05)
        probs = [[p_in, p_out], [p_out, p_in]]
        sbm = nx.stochastic_block_model(sizes, probs, seed=rng)
        regenerated.append(sbm)
    return regenerated


def _write_json(file_path: Path, records: list[dict]) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    logging.set_verbosity(logging.INFO)

    rng = np.random.RandomState(_RANDOM_SEED.value)

    algorithms = (
        ["er", "ba", "sbm", "sfn", "complete", "star", "path"]
        if _ALGORITHM.value == "all"
        else [_ALGORITHM.value]
    )

    # Load graphs per algorithm & split
    graphs: list[nx.Graph] = []
    algs_for_graph: list[str] = []
    for alg in algorithms:
        loaded = utils.load_graphs(_GRAPHS_DIR.value, alg, _SPLIT.value)
        graphs.extend(loaded)
        algs_for_graph.extend([alg] * len(loaded))
        logging.info("Loaded %d graph(s) for algorithm=%s split=%s", len(loaded), alg, _SPLIT.value)

    if not graphs:
        raise app.UsageError(
            f"No graphs found in {_GRAPHS_DIR.value} for algorithms={algorithms} split={_SPLIT.value}"
        )

    # Instantiate task
    TaskCls = TASK_CLASS[_TASK.value]
    task = TaskCls()

    # Special case: node_classification requires SBM regeneration
    if _TASK.value == "node_classification":
        graphs = _regenerate_sbm_graphs_like(graphs, rng)

    out_root = Path(_TASK_DIR.value) / _TASK.value
    _ensure_dir(out_root)

    total_graphs = 0
    total_samples = 0

    for idx, (g, alg) in enumerate(zip(graphs, algs_for_graph)):
        graph_id = _graph_id_from_index(alg, _SPLIT.value, idx)
        token_map = task.tokenize_graph(g, graph_id)
        samples = token_map[graph_id]

        # Minimal JSON structure
        records = [{"graph_id": graph_id, "text": sample} for sample in samples]

        out_dir = out_root / alg / _SPLIT.value
        _ensure_dir(out_dir)
        out_fp = out_dir / f"{graph_id}.json"
        _write_json(out_fp, records)

        total_graphs += 1
        total_samples += len(records)

    logging.info(
        "Task=%s | Split=%s | Algorithms=%s | %d graphs | %d samples | Output=%s",
        _TASK.value,
        _SPLIT.value,
        algorithms,
        total_graphs,
        total_samples,
        out_root,
    )


if __name__ == "__main__":
    app.run(main)
