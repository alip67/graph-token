# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Random graph generation."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import networkx as nx

from graph_generator_utils import generate_graphs

_ALGORITHM = flags.DEFINE_string(
    "algorithm",
    None,
    "The graph generating algorithm to use.",
    required=True,
)
_NUMBER_OF_GRAPHS = flags.DEFINE_integer(
    "number_of_graphs",
    None,
    "The number of graphs to generate.",
    required=True,
)
_DIRECTED = flags.DEFINE_bool(
    "directed", False, "Whether to generate directed graphs."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", None, "The output path to write the graphs.", required=True
)
_SPLIT = flags.DEFINE_string(
    "split", None, "The dataset split to generate.", required=True
)
_MIN_SPARSITY = flags.DEFINE_float("min_sparsity", 0.0, "The minimum sparsity.")
_MAX_SPARSITY = flags.DEFINE_float("max_sparsity", 1.0, "The maximum sparsity.")


def write_graphs(graphs, output_dir):
  """Writes graphs to output_dir."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  for ind, graph in enumerate(graphs):
    output_file = os.path.join(output_dir, f"{ind}.graphml")
    with open(output_file, "wb") as f:
      nx.write_graphml(graph, f)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _SPLIT.value == "train":
    random_seed = 9876
  elif _SPLIT.value == "test":
    random_seed = 1234
  elif _SPLIT.value == "valid":
    random_seed = 5432
  else:
    raise NotImplementedError(f"Unknown split: {_SPLIT.value}")

  generated_graphs = generate_graphs(
      number_of_graphs=_NUMBER_OF_GRAPHS.value,
      algorithm=_ALGORITHM.value,
      directed=_DIRECTED.value,
      random_seed=random_seed,
      er_min_sparsity=_MIN_SPARSITY.value,
      er_max_sparsity=_MAX_SPARSITY.value,
  )

  output_dir = os.path.join(
      _OUTPUT_PATH.value,
      _ALGORITHM.value,
      _SPLIT.value,
  )
  write_graphs(graphs=generated_graphs, output_dir=output_dir)


if __name__ == "__main__":
  app.run(main)
