#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PACKAGE_ROOT/Scripts/cuda-host-cxx.sh"
source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/abslayer-cuda-port.XXXXXX")"
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

FAKE_PACKAGE="$TEST_ROOT/package"
mkdir -p "$FAKE_PACKAGE/.build"

(
  unset ABSLAYER_CUDA_SCRATCH_PATH
  abslayer_configure_swiftpm_scratch "$FAKE_PACKAGE" Linux
  [[ "$ABSLAYER_SWIFTPM_SCRATCH_PATH" == "$FAKE_PACKAGE/.build-cuda-4090" ]]
  [[ "${ABSLAYER_SWIFT_BUILD_SCRATCH_ARGS[*]}" == "--scratch-path $FAKE_PACKAGE/.build-cuda-4090" ]]
  [[ "${ABSLAYER_SWIFT_PACKAGE_SCRATCH_ARGS[*]}" == "--scratch-path $FAKE_PACKAGE/.build-cuda-4090" ]]
)

(
  ABSLAYER_CUDA_SCRATCH_PATH="isolated-release"
  abslayer_configure_swiftpm_scratch "$FAKE_PACKAGE" Linux
  [[ "$ABSLAYER_SWIFTPM_SCRATCH_PATH" == "$FAKE_PACKAGE/isolated-release" ]]
)

ABSOLUTE_SCRATCH="$TEST_ROOT/absolute-release"
(
  ABSLAYER_CUDA_SCRATCH_PATH="$ABSOLUTE_SCRATCH"
  abslayer_configure_swiftpm_scratch "$FAKE_PACKAGE" Linux
  [[ "$ABSLAYER_SWIFTPM_SCRATCH_PATH" == "$ABSOLUTE_SCRATCH" ]]
)

(
  ABSLAYER_CUDA_SCRATCH_PATH="$ABSOLUTE_SCRATCH"
  abslayer_configure_swiftpm_scratch "$FAKE_PACKAGE" Darwin
  [[ "$ABSLAYER_SWIFTPM_SCRATCH_PATH" == "$FAKE_PACKAGE/.build" ]]
)

for unsafe_path in / /tmp /var/tmp "$FAKE_PACKAGE" "$FAKE_PACKAGE/.build"; do
  if (
    ABSLAYER_CUDA_SCRATCH_PATH="$unsafe_path"
    abslayer_configure_swiftpm_scratch "$FAKE_PACKAGE" Linux
  ) >/dev/null 2>&1; then
    echo "unsafe CUDA scratch path unexpectedly passed: $unsafe_path" >&2
    exit 1
  fi
done

ln -s "$FAKE_PACKAGE/.build" "$TEST_ROOT/colliding-symlink"
if (
  ABSLAYER_CUDA_SCRATCH_PATH="$TEST_ROOT/colliding-symlink"
  abslayer_configure_swiftpm_scratch "$FAKE_PACKAGE" Linux
) >/dev/null 2>&1; then
  echo "scratch symlink resolving to .build unexpectedly passed" >&2
  exit 1
fi

write_clang_fixture() {
  local path="$1"
  local label="$2"
  local major="$3"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [[ " $* " == *" -dM -E -x c++ /dev/null "* ]]; then' \
    '  echo "#define __clang__ 1"' \
    "  echo '#define __clang_major__ $major'" \
    '  exit 0' \
    'fi' \
    "echo '$label clang version $major.0.0'" > "$path"
  chmod 0755 "$path"
}

PRIMARY_CXX="$TEST_ROOT/clang++-18-primary"
ALIAS_CXX="$TEST_ROOT/clang++-20-alias"
TOO_NEW_CXX="$TEST_ROOT/clang++-21"
TOO_OLD_CXX="$TEST_ROOT/clang++-17"
GXX_CXX="$TEST_ROOT/g++-13"
NONEXECUTABLE_CXX="$TEST_ROOT/clang++-nonexec"
write_clang_fixture "$PRIMARY_CXX" primary 18
write_clang_fixture "$ALIAS_CXX" alias 20
write_clang_fixture "$TOO_NEW_CXX" too-new 21
write_clang_fixture "$TOO_OLD_CXX" too-old 17
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'if [[ " $* " == *" -dM -E -x c++ /dev/null "* ]]; then' \
  '  echo "#define __GNUC__ 13"' \
  '  exit 0' \
  'fi' \
  'echo "g++ (Ubuntu 13.3.0) 13.3.0"' > "$GXX_CXX"
chmod 0755 "$GXX_CXX"
printf '%s\n' '#!/usr/bin/env bash' 'echo nonexec' > "$NONEXECUTABLE_CXX"
chmod 0644 "$NONEXECUTABLE_CXX"

(
  unset CUDAHOSTCXX
  MLX_CUDA_HOST_CXX="$PRIMARY_CXX"
  abslayer_resolve_cuda_host_cxx
  [[ "$ABSLAYER_CUDA_HOST_CXX_SOURCE" == "MLX_CUDA_HOST_CXX" ]]
  [[ "$ABSLAYER_CUDA_HOST_CXX_PATH" == "$(readlink -f "$PRIMARY_CXX")" ]]
  [[ "$MLX_CUDA_HOST_CXX" == "$ABSLAYER_CUDA_HOST_CXX_PATH" ]]
  [[ "$ABSLAYER_CUDA_HOST_CXX_FAMILY" == "clang" ]]
  [[ "$ABSLAYER_CUDA_HOST_CXX_MAJOR" == "18" ]]
  [[ -n "$ABSLAYER_CUDA_HOST_CXX_VERSION_FINGERPRINT" ]]
)

(
  unset MLX_CUDA_HOST_CXX
  CUDAHOSTCXX="$ALIAS_CXX"
  abslayer_resolve_cuda_host_cxx
  [[ "$ABSLAYER_CUDA_HOST_CXX_SOURCE" == "CUDAHOSTCXX" ]]
  [[ "$MLX_CUDA_HOST_CXX" == "$(readlink -f "$ALIAS_CXX")" ]]
  [[ "$ABSLAYER_CUDA_HOST_CXX_MAJOR" == "20" ]]
)

(
  MLX_CUDA_HOST_CXX="$PRIMARY_CXX"
  CUDAHOSTCXX="$ALIAS_CXX"
  abslayer_resolve_cuda_host_cxx
  [[ "$ABSLAYER_CUDA_HOST_CXX_SOURCE" == "MLX_CUDA_HOST_CXX" ]]
  [[ "$ABSLAYER_CUDA_HOST_CXX_PATH" == "$(readlink -f "$PRIMARY_CXX")" ]]
)

for invalid_path in "" relative/clang++ "$TEST_ROOT/missing-clang++" "$NONEXECUTABLE_CXX"; do
  if (
    MLX_CUDA_HOST_CXX="$invalid_path"
    unset CUDAHOSTCXX
    abslayer_resolve_cuda_host_cxx
  ) >/dev/null 2>&1; then
    echo "invalid CUDA host compiler unexpectedly passed: '$invalid_path'" >&2
    exit 1
  fi
done

# An explicitly empty high-priority variable is a configuration error; it must
# not silently fall through to a valid lower-priority CUDAHOSTCXX value.
if (
  MLX_CUDA_HOST_CXX=""
  CUDAHOSTCXX="$ALIAS_CXX"
  abslayer_resolve_cuda_host_cxx
) >/dev/null 2>&1; then
  echo "empty higher-priority CUDA host compiler unexpectedly fell back" >&2
  exit 1
fi

for unsupported_compiler in "$GXX_CXX" "$TOO_OLD_CXX" "$TOO_NEW_CXX"; do
  if (
    MLX_CUDA_HOST_CXX="$unsupported_compiler"
    unset CUDAHOSTCXX
    abslayer_resolve_cuda_host_cxx
  ) >/dev/null 2>&1; then
    echo "unsupported CUDA host compiler unexpectedly passed: $unsupported_compiler" >&2
    exit 1
  fi
done

if [[ -x /usr/bin/clang++-18 ]]; then
  (
    unset MLX_CUDA_HOST_CXX CUDAHOSTCXX
    abslayer_resolve_cuda_host_cxx
    [[ "$ABSLAYER_CUDA_HOST_CXX_SOURCE" == "linux-default" ]]
    [[ "$ABSLAYER_CUDA_HOST_CXX_PATH" == "$(readlink -f /usr/bin/clang++-18)" ]]
    [[ "$ABSLAYER_CUDA_HOST_CXX_MAJOR" == "18" ]]
  )
fi

PLUGIN_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-host-cxx.patch"
PROJECTION_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma4-attention-projection-io.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"
BUILD_SCRIPT="$PACKAGE_ROOT/build-cuda-4090.sh"

git apply --numstat "$PROJECTION_PATCH" >/dev/null
grep -Fq 'environment["MLX_CUDA_HOST_CXX"]' "$PLUGIN_PATCH"
grep -Fq 'environment["CUDAHOSTCXX"]' "$PLUGIN_PATCH"
grep -Fq 'configuredCompiler.path.hasPrefix("/")' "$PLUGIN_PATCH"
grep -Fq 'FileManager.default.isExecutableFile' "$PLUGIN_PATCH"
[[ "$(grep -F -- '"--clangpp", hostCXXPath' "$PLUGIN_PATCH" | grep -c '^+')" -eq 2 ]]
grep -Fq 'MLX_SWIFT_EXPECTED_REVISION="0bb916c67f4b9e5c682cbe02a42c701c93ab5021"' "$PREPARE_SCRIPT"
grep -Fq 'mlx-swift-cuda-host-cxx.patch' "$PREPARE_SCRIPT"
grep -Fq 'PROFILE_MARKER="$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-build-profile"' "$BUILD_SCRIPT"
grep -Fq 'CACHE_PROFILE_MARKER="$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-cache-profile"' "$BUILD_SCRIPT"
grep -Fq 'mv -f "$CACHE_PROFILE_TEMP" "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT"
grep -Fq 'mv -f "$SUCCESS_PROFILE_TEMP" "$PROFILE_MARKER"' "$BUILD_SCRIPT"
grep -Fq 'host-cxx=$ABSLAYER_CUDA_HOST_CXX_PATH' "$BUILD_SCRIPT"
grep -Fq 'host-cxx-family=$ABSLAYER_CUDA_HOST_CXX_FAMILY' "$BUILD_SCRIPT"
grep -Fq 'host-cxx-major=$ABSLAYER_CUDA_HOST_CXX_MAJOR' "$BUILD_SCRIPT"
grep -Fq 'host-cxx-version=$ABSLAYER_CUDA_HOST_CXX_VERSION_FINGERPRINT' "$BUILD_SCRIPT"
grep -Fq 'SWIFT_BUILD_ARGS=(' "$BUILD_SCRIPT"
grep -Fq -- '-c release' "$BUILD_SCRIPT"
grep -Fq 'if [[ "$BIN_DIR" != */release ]]' "$BUILD_SCRIPT"
grep -Fq 'RELEASE_EXECUTABLES=(' "$BUILD_SCRIPT"
grep -Fq '.build-cuda-4090/' "$PACKAGE_ROOT/.gitignore"
grep -Fq '"--exclude=/.build-cuda-4090/***"' \
  "$PACKAGE_ROOT/Scripts/sync_remote_cuda_bundle.sh"

# Reproduce the second-stage failure independently of CUDA/glibc and prove why
# the generated source must contain the compatible (typedef-expanded) forms.
if command -v clang++ >/dev/null 2>&1; then
  RAW_FLOAT_SOURCE="$TEST_ROOT/gcc13-generated.cpp"
  COMPAT_FLOAT_SOURCE="$TEST_ROOT/clang18-generated.cpp"
  printf '%s\n' \
    'extern _Float32 f32;' \
    'extern _Float64 f64;' \
    'extern _Float128 f128;' \
    'extern _Float32x f32x;' \
    'extern _Float64x f64x;' > "$RAW_FLOAT_SOURCE"
  printf '%s\n' \
    'extern float f32;' \
    'extern double f64;' \
    'extern __float128 f128;' \
    'extern double f32x;' \
    'extern long double f64x;' > "$COMPAT_FLOAT_SOURCE"

  if clang++ --target=x86_64-unknown-linux-gnu -std=gnu++20 -fsyntax-only \
    "$RAW_FLOAT_SOURCE" >/dev/null 2>&1; then
    echo "Clang unexpectedly accepted GCC 13 native _FloatN output" >&2
    exit 1
  fi
  clang++ --target=x86_64-unknown-linux-gnu -std=gnu++20 -fsyntax-only \
    "$COMPAT_FLOAT_SOURCE"
fi

echo "ABSlayer CUDA host-compiler and scratch checks passed"
