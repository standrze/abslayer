#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: TrainingLauncherCUDAGraphContractTests.sh TRAIN_LAUNCHER" >&2
  exit 2
fi

TRAIN_LAUNCHER="$1"
[[ -f "$TRAIN_LAUNCHER" ]] || {
  echo "training launcher does not exist: $TRAIN_LAUNCHER" >&2
  exit 2
}

[[ "$(grep -F -c 'export MLX_USE_CUDA_GRAPHS=0' "$TRAIN_LAUNCHER")" -eq 1 ]]
[[ "$(grep -F -c "printf 'mlx_use_cuda_graphs=%s\\n' \"\$MLX_USE_CUDA_GRAPHS\"" \
  "$TRAIN_LAUNCHER")" -eq 1 ]]

export_line="$(grep -nF 'export MLX_USE_CUDA_GRAPHS=0' "$TRAIN_LAUNCHER" | cut -d: -f1)"
launch_line="$(grep -nF '"$train_binary" "$model_folder" "$train_json"' \
  "$TRAIN_LAUNCHER" | cut -d: -f1)"
record_line="$(grep -nF "printf 'mlx_use_cuda_graphs=%s\\n' \"\$MLX_USE_CUDA_GRAPHS\"" \
  "$TRAIN_LAUNCHER" | cut -d: -f1)"
for line in "$export_line" "$launch_line" "$record_line"; do
  [[ "$line" =~ ^[0-9]+$ ]]
done
(( export_line < launch_line ))
(( launch_line < record_line ))

echo "ABSlayer training launcher CUDA graph-off contract checks passed"
