#!/bin/bash
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate walkthetalk

MODEL_TESTED="Qwen3_8B"
MODEL_BASE="Llama3.3_70B"
TASK="WalkTheTalk"
EXAMPLE_INDICES="1"
MODE="mean"

python -u get_faithfulness.py --mode $MODE --example_indices $EXAMPLE_INDICES --task $TASK --responses_original_dir model_responses/$MODEL_BASE/$MODEL_TESTED --responses_counterfactual_dir model_responses/$MODEL_BASE/$MODEL_TESTED --implied_concepts_dir implied_concepts/$MODEL_BASE/$MODEL_TESTED --output_path ${MODEL_BASE}_${MODEL_TESTED}_${TASK}.jsonl