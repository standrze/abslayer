#!/usr/bin/env bash
set -euo pipefail

# `swift build` does not compile Metal source files. MLX looks for
# `mlx.metallib` beside the executable, so build it there explicitly.
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

"$ROOT/prepare-dependencies.sh"

BIN_DIR="$(swift build --show-bin-path)"
SHADER_ROOT="$ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated"
AIR_DIR="$ROOT/.build/metal"

if ! xcrun -sdk macosx --find metal >/dev/null 2>&1; then
  echo "The Metal compiler is missing. Install it once with:"
  echo "  xcodebuild -downloadComponent MetalToolchain"
  exit 1
fi

mkdir -p "$AIR_DIR"

SOURCES=(
  "$SHADER_ROOT/metal/steel/attn/kernels/steel_attention.metal"
  "$SHADER_ROOT/metal/arg_reduce.metal"
  "$SHADER_ROOT/metal/conv.metal"
  "$SHADER_ROOT/metal/rms_norm.metal"
  "$SHADER_ROOT/metal/random.metal"
  "$SHADER_ROOT/metal/scaled_dot_product_attention.metal"
  "$SHADER_ROOT/metal/gemv.metal"
  "$SHADER_ROOT/metal/layer_norm.metal"
  "$SHADER_ROOT/metal/rope.metal"
)

for source in "${SOURCES[@]}"; do
  name="$(basename "$source" .metal)"
  xcrun -sdk macosx metal \
    -std=metal4.0 \
    -Wno-c++20-extensions \
    -I "$SHADER_ROOT" \
    -c "$source" \
    -o "$AIR_DIR/$name.air"
done

xcrun -sdk macosx metallib "$AIR_DIR"/*.air -o "$BIN_DIR/mlx.metallib"

# SwiftPM test executables live inside an `.xctest` bundle rather than directly
# in BIN_DIR. Copy into every already-built bundle without depending on the
# package name (which is intentionally free to change).
while IFS= read -r -d '' test_bin; do
  cp "$BIN_DIR/mlx.metallib" "$test_bin/mlx.metallib"
done < <(find "$BIN_DIR" -type d -path '*PackageTests.xctest/Contents/MacOS' -print0)

echo "Built $BIN_DIR/mlx.metallib"
