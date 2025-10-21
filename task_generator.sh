#!/bin/bash
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

set -e
set -x

# Create and activate virtual environment
python3 -m venv graphenv
source graphenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set output paths
GRAPHS_DIR="graphs"
TASK_DIR="tasks"
TASKS=("edge_existence" "node_degree" "node_count" "edge_count" "cycle_check" "connected_nodes")

# For experimenting with only Erdos–Rényi graphs, use `er`.
# For all graph generators, set to `all`.
ALGORITHM="er"

echo "The output path is set to: $TASK_DIR"

# Loop through tasks and generate examples
for task in "${TASKS[@]}"
do
  echo "Generating examples for task $task"
  python graph_task_generator.py \
      --task="$task" \
      --algorithm="$ALGORITHM" \
      --task_dir="$TASK_DIR" \
      --graphs_dir="$GRAPHS_DIR" \
      --split=train \
      --random_seed=1234
done
