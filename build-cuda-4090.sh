#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The RTX 4090 CUDA build requires Linux." >&2
  exit 1
fi

source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
abslayer_configure_swiftpm_scratch "$PACKAGE_ROOT" Linux

SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-1}"
if [[ ! "$SWIFT_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SWIFT_BUILD_JOBS must be a positive integer." >&2
  exit 2
fi
SWIFT_BUILD_ARGS=(
  -c release
  --jobs "$SWIFT_BUILD_JOBS"
  "${ABSLAYER_SWIFT_BUILD_SCRATCH_ARGS[@]}"
)

if [[ "${SPM_CUDA:-1}" != "1" ]]; then
  echo "build-cuda-4090.sh requires SPM_CUDA=1." >&2
  exit 2
fi
export SPM_CUDA=1

if [[ "${CUDA_ARCH:-sm_89}" != "sm_89" ]]; then
  echo "The RTX 4090 wrapper requires CUDA_ARCH=sm_89." >&2
  exit 2
fi
export CUDA_ARCH=sm_89

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  # Prefer the toolkit whose headers and libraries are validated below. Ubuntu
  # may also install an older /usr/bin/nvcc, which is incompatible with those
  # headers and with the pinned Clang host compiler.
  export PATH="/usr/local/cuda/bin:$PATH"
elif ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found; install CUDA under /usr/local/cuda." >&2
  exit 1
fi

if [[ ! -d /usr/local/cuda/include || ! -d /usr/local/cuda/lib64 ]]; then
  echo "MLX Swift expects CUDA headers and libraries under /usr/local/cuda." >&2
  exit 1
fi

# CUDA 13 packages libcudacxx below include/cccl. MLX includes <cuda/cmath>,
# so make that compatibility root explicit instead of allowing a distro CUDA
# include directory to win by accident.
CUDA_CCCL_INCLUDE_DIR="$(readlink -f /usr/local/cuda/include/cccl)"
if [[ ! -f "$CUDA_CCCL_INCLUDE_DIR/cuda/cmath" ]]; then
  echo "CUDA C++ headers were not found at $CUDA_CCCL_INCLUDE_DIR/cuda/cmath." >&2
  exit 1
fi

NVCC_GPU_CODES="$(nvcc --list-gpu-code)"
if [[ "$NVCC_GPU_CODES" != *"sm_89"* ]]; then
  echo "This CUDA toolkit cannot compile sm_89 for the RTX 4090." >&2
  exit 1
fi

if [[ -n "${CUDNN_FRONTEND_INCLUDE_DIR:-}" ]]; then
  CUDNN_FRONTEND_INCLUDE_DIR="$CUDNN_FRONTEND_INCLUDE_DIR"
  CUDNN_FRONTEND_ROOT="${CUDNN_FRONTEND_ROOT:-$(dirname "$CUDNN_FRONTEND_INCLUDE_DIR")}"
else
  CUDNN_FRONTEND_ROOT="${CUDNN_FRONTEND_ROOT:-${HOME}/.local/opt/cudnn-frontend-v1.16.0}"
  CUDNN_FRONTEND_INCLUDE_DIR="$CUDNN_FRONTEND_ROOT/include"
fi
if [[ ! -f "$CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h" ]]; then
  echo "NVIDIA cudnn-frontend v1.16.0 was not found." >&2
  echo "Expected: $CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h" >&2
  echo "Install v1.16.0 there or set CUDNN_FRONTEND_ROOT/CUDNN_FRONTEND_INCLUDE_DIR." >&2
  exit 1
fi

# The default install path is versioned. Custom paths must carry verifiable
# release metadata so an older distro header cannot silently win.
CUDNN_FRONTEND_VERSION_OK=0
if [[ "$CUDNN_FRONTEND_ROOT" == *"cudnn-frontend-v1.16.0"* ]]; then
  CUDNN_FRONTEND_VERSION_OK=1
elif [[ -f "$CUDNN_FRONTEND_ROOT/.abslayer-version" ]] \
  && [[ "$(<"$CUDNN_FRONTEND_ROOT/.abslayer-version")" == "1.16.0" ]]; then
  CUDNN_FRONTEND_VERSION_OK=1
elif [[ -f "$CUDNN_FRONTEND_ROOT/CMakeLists.txt" ]] \
  && grep -Eq 'VERSION[[:space:]]+1\.16\.0' "$CUDNN_FRONTEND_ROOT/CMakeLists.txt"; then
  CUDNN_FRONTEND_VERSION_OK=1
elif [[ -d "$CUDNN_FRONTEND_ROOT/.git" ]] \
  && git -C "$CUDNN_FRONTEND_ROOT" describe --tags --exact-match HEAD 2>/dev/null \
    | grep -Eq '^v?1\.16\.0$'; then
  CUDNN_FRONTEND_VERSION_OK=1
fi
if [[ "$CUDNN_FRONTEND_VERSION_OK" != "1" ]]; then
  echo "Could not verify cudnn-frontend v1.16.0 at $CUDNN_FRONTEND_ROOT." >&2
  echo "Use the versioned default path or add .abslayer-version containing 1.16.0." >&2
  exit 1
fi

if [[ -n "${CUTLASS_INCLUDE_DIR:-}" ]]; then
  CUTLASS_INCLUDE_DIR="$CUTLASS_INCLUDE_DIR"
  CUTLASS_ROOT="${CUTLASS_ROOT:-$(dirname "$CUTLASS_INCLUDE_DIR")}"
else
  CUTLASS_ROOT="${CUTLASS_ROOT:-${HOME}/.local/opt/cutlass-v4.3.5}"
  CUTLASS_INCLUDE_DIR="$CUTLASS_ROOT/include"
fi

for CUTLASS_HEADER in \
  cutlass/cutlass.h \
  cutlass/version.h \
  cute/tensor.hpp
do
  if [[ ! -f "$CUTLASS_INCLUDE_DIR/$CUTLASS_HEADER" ]]; then
    echo "NVIDIA CUTLASS v4.3.5 (including CuTe) was not found." >&2
    echo "Expected: $CUTLASS_INCLUDE_DIR/$CUTLASS_HEADER" >&2
    echo "Install v4.3.5 there or set CUTLASS_ROOT/CUTLASS_INCLUDE_DIR." >&2
    exit 1
  fi
done

# CUTLASS carries its release in a public header. Validate the header itself,
# rather than trusting a directory name, so a stale or partial checkout cannot
# silently satisfy SwiftPM's CUDA plugin include search.
CUTLASS_VERSION_HEADER="$CUTLASS_INCLUDE_DIR/cutlass/version.h"
if ! grep -Eq '^[[:space:]]*#define[[:space:]]+CUTLASS_MAJOR[[:space:]]+4([[:space:]]|$)' \
    "$CUTLASS_VERSION_HEADER" \
  || ! grep -Eq '^[[:space:]]*#define[[:space:]]+CUTLASS_MINOR[[:space:]]+3([[:space:]]|$)' \
    "$CUTLASS_VERSION_HEADER" \
  || ! grep -Eq '^[[:space:]]*#define[[:space:]]+CUTLASS_PATCH[[:space:]]+5([[:space:]]|$)' \
    "$CUTLASS_VERSION_HEADER"
then
  echo "Could not verify CUTLASS v4.3.5 at $CUTLASS_ROOT." >&2
  echo "The MLX CUDA backend is pinned to NVIDIA CUTLASS tag v4.3.5." >&2
  exit 1
fi

CUDNN_FRONTEND_INCLUDE_DIR="$(readlink -f "$CUDNN_FRONTEND_INCLUDE_DIR")"
CUTLASS_INCLUDE_DIR="$(readlink -f "$CUTLASS_INCLUDE_DIR")"
export CPATH="$CUDA_CCCL_INCLUDE_DIR:$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR${CPATH:+:$CPATH}"

# CUDA 13 rejects Swift 6.3's bundled Clang 21 as an nvcc host compiler.
# GCC 13 is also unsafe here because its generated host C++ contains glibc
# _FloatN spellings that SwiftPM's Clang 21 cannot compile in the next phase.
# Resolve one Clang 18-20 path and use it consistently for compile and link.
source "$PACKAGE_ROOT/Scripts/cuda-host-cxx.sh"
abslayer_resolve_cuda_host_cxx

cd "$PACKAGE_ROOT"
"$PACKAGE_ROOT/prepare-dependencies.sh"

NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
CUDA_TOOLKIT_PATH="$(readlink -f /usr/local/cuda)"
NVCC_VERSION_FINGERPRINT="$(nvcc --version | cksum | awk '{print $1 "-" $2}')"
CUTLASS_VERSION_FINGERPRINT="$(cksum "$CUTLASS_VERSION_HEADER" | awk '{print $1 "-" $2}')"
BUILD_PROFILE="cuda:sm_89:scratch=$ABSLAYER_SWIFTPM_SCRATCH_PATH:nvcc=$NVCC_PATH:nvcc-version=$NVCC_VERSION_FINGERPRINT:toolkit=$CUDA_TOOLKIT_PATH:cccl=$CUDA_CCCL_INCLUDE_DIR:host-cxx-source=$ABSLAYER_CUDA_HOST_CXX_SOURCE:host-cxx=$ABSLAYER_CUDA_HOST_CXX_PATH:host-cxx-family=$ABSLAYER_CUDA_HOST_CXX_FAMILY:host-cxx-major=$ABSLAYER_CUDA_HOST_CXX_MAJOR:host-cxx-version=$ABSLAYER_CUDA_HOST_CXX_VERSION_FINGERPRINT:cudnn-frontend=$CUDNN_FRONTEND_INCLUDE_DIR:cutlass=$CUTLASS_INCLUDE_DIR:cutlass-version=$CUTLASS_VERSION_FINGERPRINT"
PROFILE_MARKER="$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-build-profile"
CACHE_PROFILE_MARKER="$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-cache-profile"
PREVIOUS_PROFILE=""
if [[ -f "$CACHE_PROFILE_MARKER" ]]; then
  IFS= read -r PREVIOUS_PROFILE < "$CACHE_PROFILE_MARKER" || true
elif [[ -f "$PROFILE_MARKER" ]]; then
  # Migrate a successful cache created before the cache/success markers were
  # split. The success marker remains authoritative only for a completed build.
  IFS= read -r PREVIOUS_PROFILE < "$PROFILE_MARKER" || true
fi

# Any build attempt can mutate the scratch tree, so its previous success marker
# is no longer authoritative. The cache marker below is deliberately separate:
# it may survive an interrupted build and allow an exact-profile resume.
rm -f "$PROFILE_MARKER"

if [[ "$PREVIOUS_PROFILE" != "$BUILD_PROFILE" ]]; then
  echo "Preparing build cache for RTX 4090 CUDA (sm_89)"
  swift package "${ABSLAYER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" clean
fi

# Record the cache identity before compiling. Write through a same-directory
# temporary file so interruption cannot leave a partial profile as valid.
mkdir -p "$(dirname "$CACHE_PROFILE_MARKER")"
CACHE_PROFILE_TEMP="$CACHE_PROFILE_MARKER.tmp.$$"
printf '%s\n' "$BUILD_PROFILE" > "$CACHE_PROFILE_TEMP"
mv -f "$CACHE_PROFILE_TEMP" "$CACHE_PROFILE_MARKER"

echo "Building ABSlayer with MLX CUDA arch=sm_89 jobs=$SWIFT_BUILD_JOBS scratch=$ABSLAYER_SWIFTPM_SCRATCH_PATH"
swift build "${SWIFT_BUILD_ARGS[@]}"

BIN_DIR="$(swift build "${SWIFT_BUILD_ARGS[@]}" --show-bin-path)"
if [[ "$BIN_DIR" != */release ]]; then
  echo "SwiftPM did not return a release binary directory: $BIN_DIR" >&2
  exit 1
fi

RELEASE_EXECUTABLES=(
  abslayer
  abslayer-cli
  abslayer-eval
  abslayer-preflight
  abslayer-kl
  abslayer-continuation-kl
  abslayer-optimize
  abslayer-intervene
  abslayer-answer-transport
  abslayer-matched-patch
  abslayer-prompt-end-patch
  abslayer-judge
  abslayer-dataset
  abslayer-prefix-train
  abslayer-lora-merge
  abslayer-request
  abslayer-lora-pipeline
  abslayer-training-data
  abslayer-rmsnorm-diagnostic
  abslayer-directional-adapter
  abslayer-preference-capture
  abslayer-first-token-screen
  abslayer-som-search
  abslayer-som-multisource-search
  abslayer-ara
)
for executable in "${RELEASE_EXECUTABLES[@]}"; do
  if [[ ! -x "$BIN_DIR/$executable" ]]; then
    echo "Expected release executable is missing: $BIN_DIR/$executable" >&2
    exit 1
  fi
done

# Only verified release executables constitute a successful build. Keep this
# marker absent on compile or verification failure, and publish it atomically.
mkdir -p "$(dirname "$PROFILE_MARKER")"
SUCCESS_PROFILE_TEMP="$PROFILE_MARKER.tmp.$$"
printf '%s\n' "$BUILD_PROFILE" > "$SUCCESS_PROFILE_TEMP"
mv -f "$SUCCESS_PROFILE_TEMP" "$PROFILE_MARKER"
echo "Built ABSlayer CUDA executables in $BIN_DIR"
