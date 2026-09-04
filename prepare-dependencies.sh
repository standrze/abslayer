#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST_OS="$(uname -s)"
source "$ROOT/Scripts/swiftpm-scratch-path.sh"
abslayer_configure_swiftpm_scratch "$ROOT" "$HOST_OS"
cd "$ROOT"

if [[ "$HOST_OS" == "Linux" ]]; then
    swift package "${ABSLAYER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" resolve
else
    swift package resolve
fi
MLX_SWIFT_CHECKOUT="$ABSLAYER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift"
MLX_SWIFT_PATCH="$ROOT/Patches/mlx-swift-cuda-host-cxx.patch"
MLX_SWIFT_EXPECTED_REVISION="0bb916c67f4b9e5c682cbe02a42c701c93ab5021"
MLX_SOURCE_CHECKOUT="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx"
MLX_SOURCE_PATCH="$ROOT/Patches/mlx-cuda-half-fmod.patch"
MLX_RMS_NORM_PATCH="$ROOT/Patches/mlx-cuda-rms-norm-small-row.patch"
MLX_SOURCE_EXPECTED_REVISION="ce45c52505c8158ea48d2a54e8caae05efd86bfe"
CHECKOUT="$ABSLAYER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift-lm"
TARGET="$CHECKOUT/Libraries/MLXLLM/Models/Gemma4Text.swift"
PATCH="$ROOT/Patches/mlx-swift-lm-gemma4-layer-states.patch"
INTERVENTION_PATCH="$ROOT/Patches/mlx-swift-lm-gemma4-residual-intervention.patch"
PROJECTION_IO_PATCH="$ROOT/Patches/mlx-swift-lm-gemma4-attention-projection-io.patch"
README_PATCH="$ROOT/Patches/mlx-swift-lm-ignore-readmes.patch"
COREFOUNDATION_PATCH="$ROOT/Patches/mlx-swift-lm-corefoundation-linux.patch"
LORA_NUMERICS_PATCH="$ROOT/Patches/mlx-swift-lm-lora-numerics.patch"
EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"

if [[ ! -d "$MLX_SWIFT_CHECKOUT" ]]; then
    echo "mlx-swift checkout was not created at $MLX_SWIFT_CHECKOUT" >&2
    exit 1
fi
if [[ ! -d "$MLX_SOURCE_CHECKOUT" ]]; then
    echo "mlx source checkout was not created at $MLX_SOURCE_CHECKOUT" >&2
    exit 1
fi
if [[ ! -d "$CHECKOUT" ]]; then
    echo "mlx-swift-lm checkout was not created at $CHECKOUT" >&2
    exit 1
fi

# Verify all pinned dependency revisions, including mlx-swift's nested MLX
# source checkout, before mutating any checkout.
MLX_SWIFT_ACTUAL_REVISION="$(git -C "$MLX_SWIFT_CHECKOUT" rev-parse HEAD)"
if [[ "$MLX_SWIFT_ACTUAL_REVISION" != "$MLX_SWIFT_EXPECTED_REVISION" ]]; then
    echo "Refusing to patch unexpected mlx-swift revision: $MLX_SWIFT_ACTUAL_REVISION" >&2
    echo "Expected: $MLX_SWIFT_EXPECTED_REVISION" >&2
    exit 1
fi
MLX_SOURCE_ACTUAL_REVISION="$(git -C "$MLX_SOURCE_CHECKOUT" rev-parse HEAD)"
if [[ "$MLX_SOURCE_ACTUAL_REVISION" != "$MLX_SOURCE_EXPECTED_REVISION" ]]; then
    echo "Refusing to patch unexpected mlx source revision: $MLX_SOURCE_ACTUAL_REVISION" >&2
    echo "Expected: $MLX_SOURCE_EXPECTED_REVISION" >&2
    exit 1
fi
ACTUAL_REVISION="$(git -C "$CHECKOUT" rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]]; then
    echo "Refusing to patch unexpected mlx-swift-lm revision: $ACTUAL_REVISION" >&2
    echo "Expected: $EXPECTED_REVISION" >&2
    exit 1
fi

if git -C "$MLX_SWIFT_CHECKOUT" apply --reverse --check "$MLX_SWIFT_PATCH" >/dev/null 2>&1; then
    echo "mlx-swift CUDA host-compiler patch already applied."
elif git -C "$MLX_SWIFT_CHECKOUT" apply --check "$MLX_SWIFT_PATCH" >/dev/null 2>&1; then
    git -C "$MLX_SWIFT_CHECKOUT" apply "$MLX_SWIFT_PATCH"
    echo "Applied mlx-swift CUDA host-compiler patch."
else
    echo "Could not apply the pinned mlx-swift CUDA host-compiler patch." >&2
    exit 1
fi

if git -C "$MLX_SOURCE_CHECKOUT" apply --reverse --check "$MLX_SOURCE_PATCH" >/dev/null 2>&1; then
    echo "mlx CUDA half/bfloat16 fmod patch already applied."
elif git -C "$MLX_SOURCE_CHECKOUT" apply --check "$MLX_SOURCE_PATCH" >/dev/null 2>&1; then
    git -C "$MLX_SOURCE_CHECKOUT" apply "$MLX_SOURCE_PATCH"
    echo "Applied mlx CUDA half/bfloat16 fmod patch."
else
    echo "Could not apply the pinned mlx CUDA half/bfloat16 fmod patch." >&2
    exit 1
fi

# Exact runtime backport of upstream MLX a5a684db596c117f13f7bacaea9902d0ad6d28a6
# (Fix CUDA RMSNorm small-row dispatch, #3792). Keep this separate from the
# checkout revision so dependency preparation remains reproducible and
# fail-closed if either the pin or upstream context changes.
if git -C "$MLX_SOURCE_CHECKOUT" apply --reverse --check "$MLX_RMS_NORM_PATCH" >/dev/null 2>&1; then
    echo "mlx CUDA RMSNorm small-row patch already applied."
elif git -C "$MLX_SOURCE_CHECKOUT" apply --check "$MLX_RMS_NORM_PATCH" >/dev/null 2>&1; then
    git -C "$MLX_SOURCE_CHECKOUT" apply "$MLX_RMS_NORM_PATCH"
    echo "Applied mlx CUDA RMSNorm small-row patch."
else
    echo "Could not apply the pinned mlx CUDA RMSNorm small-row patch." >&2
    exit 1
fi

chmod u+w "$TARGET"
if grep -Fq 'public func layerHiddenStates' "$TARGET" \
    && grep -Fq 'states.reserveCapacity(config.numHiddenLayers)' "$TARGET" \
    && grep -A2 -F 'public var loraLayers' "$TARGET" | grep -Fq 'model.layers'; then
    echo "Gemma 4 layer-state patch already applied."
elif git -C "$CHECKOUT" apply --check "$PATCH" >/dev/null 2>&1; then
    git -C "$CHECKOUT" apply "$PATCH"
    echo "Applied Gemma 4 layer-state patch."
else
    echo "Could not apply the pinned Gemma 4 layer-state patch." >&2
    exit 1
fi

if grep -Fq 'public var residualIntervention' "$TARGET" \
    && grep -Fq 'residualIntervention?(idx, out)' "$TARGET" \
    && grep -Fq 'residualIntervention?(index, out)' "$TARGET"; then
    echo "Gemma 4 residual-intervention patch already applied."
elif git -C "$CHECKOUT" apply --check "$INTERVENTION_PATCH" >/dev/null 2>&1; then
    git -C "$CHECKOUT" apply "$INTERVENTION_PATCH"
    echo "Applied Gemma 4 residual-intervention patch."
else
    echo "Could not apply the pinned Gemma 4 residual-intervention patch." >&2
    exit 1
fi

if grep -Fq 'var outputProjectionTap' "$TARGET" \
    && grep -Fq 'public func attentionOutputProjectionIO' "$TARGET"; then
    echo "Gemma 4 attention-projection I/O patch already applied."
elif git -C "$CHECKOUT" apply --check "$PROJECTION_IO_PATCH" >/dev/null 2>&1; then
    git -C "$CHECKOUT" apply "$PROJECTION_IO_PATCH"
    echo "Applied Gemma 4 attention-projection I/O patch."
else
    echo "Could not apply the pinned Gemma 4 attention-projection I/O patch." >&2
    exit 1
fi

if git -C "$CHECKOUT" apply --reverse --check "$README_PATCH" >/dev/null 2>&1; then
    echo "mlx-swift-lm README exclusions already applied."
elif git -C "$CHECKOUT" apply --check "$README_PATCH" >/dev/null 2>&1; then
    git -C "$CHECKOUT" apply "$README_PATCH"
    echo "Applied mlx-swift-lm README exclusions."
else
    echo "Could not apply the pinned mlx-swift-lm README exclusions." >&2
    exit 1
fi

if git -C "$CHECKOUT" apply --reverse --check "$COREFOUNDATION_PATCH" >/dev/null 2>&1; then
    echo "mlx-swift-lm CoreFoundation import already applied."
elif git -C "$CHECKOUT" apply --check "$COREFOUNDATION_PATCH" >/dev/null 2>&1; then
    git -C "$CHECKOUT" apply "$COREFOUNDATION_PATCH"
    echo "Applied mlx-swift-lm CoreFoundation import."
else
    echo "Could not apply the pinned mlx-swift-lm CoreFoundation import." >&2
    exit 1
fi

if git -C "$CHECKOUT" apply --reverse --check "$LORA_NUMERICS_PATCH" >/dev/null 2>&1; then
    echo "mlx-swift-lm LoRA numerical-identity patch already applied."
elif git -C "$CHECKOUT" apply --check "$LORA_NUMERICS_PATCH" >/dev/null 2>&1; then
    git -C "$CHECKOUT" apply "$LORA_NUMERICS_PATCH"
    echo "Applied mlx-swift-lm LoRA numerical-identity patch."
else
    echo "Could not apply the pinned mlx-swift-lm LoRA numerical-identity patch." >&2
    exit 1
fi
