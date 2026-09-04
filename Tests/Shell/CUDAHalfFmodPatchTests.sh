#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-cuda-half-fmod.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'MLX_SOURCE_EXPECTED_REVISION="ce45c52505c8158ea48d2a54e8caae05efd86bfe"' \
  "$PREPARE_SCRIPT"
grep -Fq 'mlx-cuda-half-fmod.patch' "$PREPARE_SCRIPT"
grep -Fq 'MLX_SOURCE_ACTUAL_REVISION="$(git -C "$MLX_SOURCE_CHECKOUT" rev-parse HEAD)"' \
  "$PREPARE_SCRIPT"
grep -Fq 'git -C "$MLX_SOURCE_CHECKOUT" apply --reverse --check "$MLX_SOURCE_PATCH"' \
  "$PREPARE_SCRIPT"

# The nested MLX revision must be validated before the first dependency is
# mutated. This preserves the all-or-nothing revision guard in preparation.
REVISION_CHECK_LINE="$(grep -n -F 'if [[ "$MLX_SOURCE_ACTUAL_REVISION" != "$MLX_SOURCE_EXPECTED_REVISION" ]]' \
  "$PREPARE_SCRIPT" | cut -d: -f1)"
FIRST_MUTATION_LINE="$(grep -n -F 'git -C "$MLX_SWIFT_CHECKOUT" apply "$MLX_SWIFT_PATCH"' \
  "$PREPARE_SCRIPT" | cut -d: -f1)"
if [[ -z "$REVISION_CHECK_LINE" || -z "$FIRST_MUTATION_LINE" \
  || "$REVISION_CHECK_LINE" -ge "$FIRST_MUTATION_LINE" ]]; then
  echo "nested MLX revision is not checked before dependency mutation" >&2
  exit 1
fi

require_added_count() {
  local expected="$1"
  local pattern="$2"
  local actual
  actual="$(grep -F "$pattern" "$PATCH_FILE" | grep -c '^+' || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected added occurrence(s) of '$pattern'; found $actual" >&2
    exit 1
  fi
}

require_added_count 1 'if constexpr (cuda::std::is_same_v<T, __half>)'
require_added_count 1 'cuda::std::is_same_v<T, __nv_bfloat16>'
require_added_count 1 'cuda::std::fmod(__half2float(x), __half2float(y))'
require_added_count 1 'cuda::std::fmod(__bfloat162float(x), __bfloat162float(y))'

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/abslayer-half-fmod.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

SOURCE_DIR="$TEST_ROOT/mlx/backend/cuda/device"
mkdir -p "$SOURCE_DIR"
cat > "$SOURCE_DIR/binary_ops.cuh" <<'EOF'
struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      if constexpr (cuda::std::is_signed_v<T>) {
        auto r = x % y;
        if (r != 0 && (r < 0 != y < 0)) {
          r += y;
        }
        return r;
      } else {
        return x % y;
      }
    } else if constexpr (is_complex_v<T>) {
      return x % y;
    } else {
      T r = cuda::std::fmod(x, y);
      if (r != 0 && (r < 0 != y < 0)) {
        r = r + y;
      }
      return r;
    }
  }
};
EOF

git -C "$TEST_ROOT" apply --check "$PATCH_FILE"
git -C "$TEST_ROOT" apply "$PATCH_FILE"
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"

if git -C "$TEST_ROOT" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "fmod patch unexpectedly remained forward-applicable" >&2
  exit 1
fi
[[ "$(grep -F -c '__half2float(x)' "$SOURCE_DIR/binary_ops.cuh")" -eq 1 ]]
[[ "$(grep -F -c '__bfloat162float(x)' "$SOURCE_DIR/binary_ops.cuh")" -eq 1 ]]
[[ "$(grep -F -c 'T r = cuda::std::fmod(x, y);' "$SOURCE_DIR/binary_ops.cuh")" -eq 0 ]]

echo "ABSlayer CUDA half/bfloat16 fmod compatibility patch checks passed"
