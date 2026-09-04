#!/usr/bin/env bash

abslayer_configure_swiftpm_scratch() {
  local package_root="$1"
  local host_os="$2"
  local requested
  local candidate
  local resolved

  ABSLAYER_SWIFTPM_SCRATCH_PATH="$package_root/.build"
  ABSLAYER_SWIFT_BUILD_SCRATCH_ARGS=()
  ABSLAYER_SWIFT_PACKAGE_SCRATCH_ARGS=()

  if [[ "$host_os" != "Linux" ]]; then
    return 0
  fi

  requested="${ABSLAYER_CUDA_SCRATCH_PATH:-$package_root/.build-cuda-4090}"
  if [[ "$requested" == /* ]]; then
    candidate="${requested%/}"
  else
    candidate="$package_root/${requested%/}"
  fi

  case "$candidate" in
    ""|/|/tmp|/var/tmp|"$package_root"|"${HOME:-}"|"$package_root/.build")
      echo "ABSLAYER_CUDA_SCRATCH_PATH must name a dedicated CUDA build directory." >&2
      echo "Refusing colliding or broad scratch path: $candidate" >&2
      return 2
      ;;
  esac

  mkdir -p "$candidate"
  resolved="$(cd "$candidate" && pwd -P)"
  case "$resolved" in
    ""|/|/tmp|/var/tmp|"$package_root"|"${HOME:-}"|"$package_root/.build")
      echo "ABSLAYER_CUDA_SCRATCH_PATH resolves to a colliding path: $resolved" >&2
      return 2
      ;;
  esac

  ABSLAYER_SWIFTPM_SCRATCH_PATH="$resolved"
  ABSLAYER_CUDA_SCRATCH_PATH="$resolved"
  ABSLAYER_SWIFT_BUILD_SCRATCH_ARGS=(--scratch-path "$resolved")
  ABSLAYER_SWIFT_PACKAGE_SCRATCH_ARGS=(--scratch-path "$resolved")
  export ABSLAYER_SWIFTPM_SCRATCH_PATH ABSLAYER_CUDA_SCRATCH_PATH
}
