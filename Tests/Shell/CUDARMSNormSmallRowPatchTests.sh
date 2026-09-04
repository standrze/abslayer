#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-cuda-rms-norm-small-row.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'MLX_SOURCE_EXPECTED_REVISION="ce45c52505c8158ea48d2a54e8caae05efd86bfe"' \
  "$PREPARE_SCRIPT"
grep -Fq 'MLX_RMS_NORM_PATCH="$ROOT/Patches/mlx-cuda-rms-norm-small-row.patch"' \
  "$PREPARE_SCRIPT"
grep -Fq 'a5a684db596c117f13f7bacaea9902d0ad6d28a6' "$PREPARE_SCRIPT"
grep -Fq 'git -C "$MLX_SOURCE_CHECKOUT" apply --reverse --check "$MLX_RMS_NORM_PATCH"' \
  "$PREPARE_SCRIPT"

revision_check_line="$(
  grep -nF 'if [[ "$MLX_SOURCE_ACTUAL_REVISION" != "$MLX_SOURCE_EXPECTED_REVISION" ]]' \
    "$PREPARE_SCRIPT" | cut -d: -f1
)"
rms_apply_line="$(
  grep -nF 'git -C "$MLX_SOURCE_CHECKOUT" apply "$MLX_RMS_NORM_PATCH"' \
    "$PREPARE_SCRIPT" | cut -d: -f1
)"
first_model_patch_line="$(
  grep -nF 'chmod u+w "$TARGET"' "$PREPARE_SCRIPT" | cut -d: -f1
)"
for line in "$revision_check_line" "$rms_apply_line" "$first_model_patch_line"; do
  [[ "$line" =~ ^[0-9]+$ ]]
done
(( revision_check_line < rms_apply_line ))
(( rms_apply_line < first_model_patch_line ))

numstat="$(git apply --numstat "$PATCH_FILE")"
[[ "$numstat" == $'3\t1\tmlx/backend/cuda/rms_norm.cu' ]]

require_patch_count() {
  local prefix="$1"
  local expected="$2"
  local pattern="$3"
  local actual
  actual="$(grep -F "$pattern" "$PATCH_FILE" | grep -c "^$prefix" || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected '$prefix$pattern' patch line(s); found $actual" >&2
    exit 1
  fi
}

require_patch_count '-' 1 '      std::integral_constant<int, 2>());'
require_patch_count '+' 1 '      std::integral_constant<int, 1>());'
require_patch_count '+' 2 '            static_assert(block_dim <= 32 || groups_per_block() == 1);'

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/abslayer-rms-norm.XXXXXX")"
cleanup() {
  local result=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$result"
}
trap cleanup EXIT

SOURCE_DIR="$TEST_ROOT/mlx/backend/cuda"
mkdir -p "$SOURCE_DIR"
cat > "$SOURCE_DIR/rms_norm.cu" <<'EOF'
template <int n_per_thread, typename F>
void dispatch_group_dim(int axis_size, F&& f) {
  if (axis_size <= n_per_thread * 32) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 1>(),
      std::integral_constant<int, 4>());
  } else if (axis_size <= n_per_thread * 32 * 2) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 2>(),
      std::integral_constant<int, 2>());
  } else if (axis_size <= n_per_thread * 32 * 4) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 4>(),
      std::integral_constant<int, 1>());
  }
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
      dispatch_group_dim<N_READS>(
          axis_size, [&](auto group_dim, auto n_groups, auto groups_per_block) {
            constexpr int block_dim = n_groups() * group_dim();
            auto kernel =
                cu::rms_norm_small<DataType, block_dim, group_dim(), N_READS>;
            auto n_blocks =
                (n_rows + groups_per_block() - 1) / groups_per_block();
          });
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
        dispatch_group_dim<N_READS>(
            axis_size,
            [&](auto group_dim, auto n_groups, auto groups_per_block) {
              constexpr int block_dim = group_dim() * n_groups();
              auto kernel = cu::rms_norm_vjp_small<
                  DataType,
                  has_w_constant.value,
                  block_dim,
                  group_dim(),
                  N_READS>;
            });
}
EOF

git -C "$TEST_ROOT" apply --check "$PATCH_FILE"
git -C "$TEST_ROOT" apply "$PATCH_FILE"
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"

if git -C "$TEST_ROOT" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "RMSNorm patch unexpectedly remained forward-applicable" >&2
  exit 1
fi
[[ "$(grep -F -c 'static_assert(block_dim <= 32 || groups_per_block() == 1);' \
  "$SOURCE_DIR/rms_norm.cu")" -eq 2 ]]
[[ "$(grep -F -c '      std::integral_constant<int, 2>());' \
  "$SOURCE_DIR/rms_norm.cu")" -eq 0 ]]
[[ "$(grep -F -c '      std::integral_constant<int, 1>());' \
  "$SOURCE_DIR/rms_norm.cu")" -eq 2 ]]

echo "ABSlayer CUDA RMSNorm small-row dispatch patch checks passed"
