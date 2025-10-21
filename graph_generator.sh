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

# Set output path
OUTPUT_PATH="graphs"
echo "The output path is set to: $OUTPUT_PATH"

# Loop through algorithms and generate graphs
for algorithm in "er" "ba" "sbm" "sfn" "complete" "star" "path"
do
  echo "Generating test examples for $algorithm"
  python3 graph_generator.py \
      --algorithm="$algorithm" \
      --number_of_graphs=500 \
      --split=train \
      --output_path="$OUTPUT_PATH"
done
