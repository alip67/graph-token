# coding=utf-8
# Copyright 2025 ...
# Licensed under the Apache License, Version 2.0

"""Utility functions for loading and saving graphs without TensorFlow."""

import os
import json
import networkx as nx


def load_graphs(base_path, algorithm, split, max_nnodes=20):
    """Load a list of graphs from a given algorithm and split.

    Args:
        base_path: Root directory where graphs/<algorithm>/<split>/ exist.
        algorithm: Graph generator algorithm name (e.g., 'er', 'ba', etc.).
        split: Dataset split ('train', 'valid', 'test').
        max_nnodes: Maximum number of nodes to include a graph.

    Returns:
        A list of networkx.Graph objects.
    """
    graphs_path = os.path.join(base_path, algorithm, split)
    loaded_graphs = []

    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"Path not found: {graphs_path}")

    for file in os.listdir(graphs_path):
        if file.endswith(".graphml"):
            path = os.path.join(graphs_path, file)
            with open(path, "rb") as f:
                graph = nx.read_graphml(f, node_type=int)
            if graph.number_of_nodes() <= max_nnodes:
                loaded_graphs.append(graph)

    return loaded_graphs


def write_examples(examples, output_path):
    """Write a list of serialized examples (e.g., dicts) as a JSON file.

    Args:
        examples: Iterable of serializable Python dicts or JSON-compatible objects.
        output_path: Destination file path for JSON output.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
