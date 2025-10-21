# coding=utf-8
# Copyright 2025 ...
# Licensed under the Apache License, Version 2.0

"""The graph tasks to be tried with LLMs."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np


def _base_tokens(graph: nx.Graph) -> List[str]:
  """Return the shared prefix tokens encoding edges and node list."""
  tokens = ["<bos>"]
  # Undirected: add each undirected edge once; Directed: keep direction.
  if isinstance(graph, nx.DiGraph):
    # Keep directions, canonicalize by (u, v) order appearance.
    for u, v in graph.edges():
      tokens.extend([str(u), str(v), "<e>"])
  else:
    added = set()
    for u, v in graph.edges():
      a, b = sorted((u, v))
      if (a, b) not in added:
        tokens.extend([str(a), str(b), "<e>"])
        added.add((a, b))

  tokens.append("<n>")
  # Use deterministic node order [0..N-1] if possible; otherwise sorted labels.
  try:
    n = graph.number_of_nodes()
    nodes = list(range(n))
    # If labels are non-contiguous integers, fallback to sorted(graph.nodes()).
    if set(nodes) != set(graph.nodes()):
      nodes = sorted(graph.nodes())
  except Exception:
    nodes = sorted(graph.nodes())
  tokens.extend([str(i) for i in nodes])
  return tokens


class GraphTask:
  """Base class for all graph tasks."""

  name: str = "default"
  maximum_nnodes_cot_graph: int = 10  # Not used yet, kept for future CoT variants.

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    """Return {graph_id: [tokenized_sample, ...]}. Override in subclasses."""
    raise NotImplementedError


# --------------------------- Concrete tasks ---------------------------

class NodeDegree(GraphTask):
  name = "node_degree"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    degrees = dict(graph.degree())
    base = _base_tokens(graph)
    samples = []
    for q in sorted(graph.nodes()):
      s = list(base)
      s.extend(["<q>", str(q)])
      s.extend(["<p>", f"d{degrees[q]}", "<eos>"])
      samples.append(" ".join(s))
    return {graph_id: samples}


class NodeCount(GraphTask):
  name = "node_count"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    n = graph.number_of_nodes()
    s = list(base)
    s.extend(["<q>", "node_count"])
    s.extend(["<p>", f"n{n}", "<eos>"])
    return {graph_id: [" ".join(s)]}


class EdgeCount(GraphTask):
  name = "edge_count"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    m = graph.number_of_edges()
    # For undirected graphs, NetworkX stores each edge once already.
    s = list(base)
    s.extend(["<q>", "edge_count"])
    s.extend(["<p>", f"m{m}", "<eos>"])
    return {graph_id: [" ".join(s)]}


class EdgeExistence(GraphTask):
  name = "edge_existence"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    nodes = sorted(graph.nodes())
    if isinstance(graph, nx.DiGraph):
      # All ordered pairs (u, v), u != v
      pairs = [(u, v) for u in nodes for v in nodes if u != v]
    else:
      # All unordered pairs
      pairs = list(itertools.combinations(nodes, 2))
    for u, v in pairs:
      s = list(base)
      s.extend(["<q>", str(u), str(v)])
      exists = graph.has_edge(u, v) if isinstance(graph, nx.DiGraph) else graph.has_edge(u, v) or graph.has_edge(v, u)
      s.extend(["<p>", "yes" if exists else "no", "<eos>"])
      samples.append(" ".join(s))
    return {graph_id: samples}


class ConnectedNodes(GraphTask):
  name = "connected_nodes"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    for u in sorted(graph.nodes()):
      s = list(base)
      s.extend(["<q>", "neighbors", str(u)])
      neighbors = sorted(graph.successors(u)) if isinstance(graph, nx.DiGraph) else sorted(graph.neighbors(u))
      # Label as a space-separated list wrapped by braces for clarity.
      label = ["{"] + [str(v) for v in neighbors] + ["}"]
      s.extend(["<p>", *label, "<eos>"])
      samples.append(" ".join(s))
    return {graph_id: samples}


class CycleCheck(GraphTask):
  name = "cycle_check"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    s = list(base)
    s.extend(["<q>", "has_cycle"])
    if isinstance(graph, nx.DiGraph):
      try:
        has_cycle = not nx.is_directed_acyclic_graph(graph)
      except nx.NetworkXError:
        has_cycle = True
    else:
      has_cycle = len(nx.cycle_basis(graph)) > 0
    s.extend(["<p>", "yes" if has_cycle else "no", "<eos>"])
    return {graph_id: [" ".join(s)]}


class DisconnectedNodes(GraphTask):
  name = "disconnected_nodes"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    isolated = sorted(list(nx.isolates(graph)))
    s = list(base)
    s.extend(["<q>", "isolated_nodes"])
    label = ["{"] + [str(v) for v in isolated] + ["}"]
    s.extend(["<p>", *label, "<eos>"])
    samples.append(" ".join(s))
    return {graph_id: samples}


class Reachability(GraphTask):
  name = "reachability"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    nodes = sorted(graph.nodes())
    if isinstance(graph, nx.DiGraph):
      pairs = [(u, v) for u in nodes for v in nodes if u != v]
      for u, v in pairs:
        s = list(base)
        s.extend(["<q>", "reachable", str(u), str(v)])
        reachable = nx.has_path(graph, u, v)
        s.extend(["<p>", "yes" if reachable else "no", "<eos>"])
        samples.append(" ".join(s))
    else:
      # Undirected: connectivity is symmetric; only use unordered pairs.
      for u, v in itertools.combinations(nodes, 2):
        s = list(base)
        s.extend(["<q>", "reachable", str(u), str(v)])
        reachable = nx.has_path(graph, u, v)
        s.extend(["<p>", "yes" if reachable else "no", "<eos>"])
        samples.append(" ".join(s))
    return {graph_id: samples}


class ShortestPath(GraphTask):
  name = "shortest_path"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    nodes = sorted(graph.nodes())
    def _add(u, v):
      s = list(base)
      s.extend(["<q>", "shortest_distance", str(u), str(v)])
      try:
        d = nx.shortest_path_length(graph, u, v)
        s.extend(["<p>", f"len{d}", "<eos>"])
      except nx.NetworkXNoPath:
        s.extend(["<p>", "INF", "<eos>"])
      samples.append(" ".join(s))
    if isinstance(graph, nx.DiGraph):
      for u in nodes:
        for v in nodes:
          if u != v:
            _add(u, v)
    else:
      for u, v in itertools.combinations(nodes, 2):
        _add(u, v)
    return {graph_id: samples}


class MaximumFlow(GraphTask):
  """Unit-capacity max-flow between all ordered pairs (u, v), u != v."""
  name = "maximum_flow"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    nodes = sorted(graph.nodes())
    # Use unit capacity. For undirected graphs, convert to DiGraph with two arcs.
    if not isinstance(graph, nx.DiGraph):
      G = nx.DiGraph()
      G.add_nodes_from(graph.nodes())
      for u, v in graph.edges():
        G.add_edge(u, v, capacity=1.0)
        G.add_edge(v, u, capacity=1.0)
    else:
      G = graph.copy()
      for u, v in G.edges():
        if "capacity" not in G[u][v]:
          G[u][v]["capacity"] = 1.0

    for u in nodes:
      for v in nodes:
        if u == v:
          continue
        s = list(base)
        s.extend(["<q>", "maxflow", str(u), str(v)])
        # Edmonds-Karp is fine for small graphs.
        flow_value, _ = nx.maximum_flow(G, u, v, capacity="capacity", flow_func=nx.algorithms.flow.edmonds_karp)
        s.extend(["<p>", f"f{int(flow_value)}", "<eos>"])
        samples.append(" ".join(s))
    return {graph_id: samples}


class TriangleCounting(GraphTask):
  name = "triangle_counting"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    if isinstance(graph, nx.DiGraph):
      # For directed graphs, count undirected triangles on underlying undirected graph.
      und = graph.to_undirected()
      tri_map = nx.triangles(und)
    else:
      tri_map = nx.triangles(graph)
    total = sum(tri_map.values()) // 3
    s = list(base)
    s.extend(["<q>", "triangle_count"])
    s.extend(["<p>", f"t{total}", "<eos>"])
    return {graph_id: [" ".join(s)]}


class NodeClassification(GraphTask):
  """Assumes an SBM with node attribute 'block' (int class)."""
  name = "node_classification"

  def tokenize_graph(self, graph: nx.Graph, graph_id: str) -> Dict[str, List[str]]:
    base = _base_tokens(graph)
    samples = []
    # Expect `graph.nodes[u]['block']` or community label.
    nodes = sorted(graph.nodes())
    # If no 'block' attribute exists, fall back to connected-component ID.
    comp_id = {}
    if "block" not in graph.nodes[nodes[0]]:
      # Assign component IDs as pseudo-classes.
      if isinstance(graph, nx.DiGraph):
        comps = list(nx.weakly_connected_components(graph))
      else:
        comps = list(nx.connected_components(graph))
      for cid, comp in enumerate(comps):
        for u in comp:
          comp_id[u] = cid

    for u in nodes:
      s = list(base)
      s.extend(["<q>", "class", str(u)])
      if "block" in graph.nodes[u]:
        label = graph.nodes[u]["block"]
      else:
        label = comp_id[u]
      s.extend(["<p>", f"c{label}", "<eos>"])
      samples.append(" ".join(s))
    return {graph_id: samples}


# # Registry (kept here for a single source of truth)
# TASK_CLASS = {
#     "edge_existence": EdgeExistence,
#     "node_degree": NodeDegree,
#     "node_count": NodeCount,
#     "edge_count": EdgeCount,
#     "connected_nodes": ConnectedNodes,
#     "cycle_check": CycleCheck,
#     "disconnected_nodes": DisconnectedNodes,
#     "reachability": Reachability,
#     "shortest_path": ShortestPath,
#     "maximum_flow": MaximumFlow,
#     "triangle_counting": TriangleCounting,
#     "node_classification": NodeClassification,
# }
