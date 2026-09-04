#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_SCRIPT="$PACKAGE_ROOT/build-cuda-4090.sh"

grep -Fq 'PROFILE_MARKER="$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-build-profile"' \
  "$BUILD_SCRIPT"
grep -Fq 'CACHE_PROFILE_MARKER="$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-cache-profile"' \
  "$BUILD_SCRIPT"
grep -Fq 'IFS= read -r PREVIOUS_PROFILE < "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT"
grep -Fq 'elif [[ -f "$PROFILE_MARKER" ]]' "$BUILD_SCRIPT"
grep -Fq 'IFS= read -r PREVIOUS_PROFILE < "$PROFILE_MARKER"' "$BUILD_SCRIPT"

success_invalidate_line="$(
  grep -nF 'rm -f "$PROFILE_MARKER"' "$BUILD_SCRIPT" | cut -d: -f1
)"
clean_line="$(
  grep -nF 'swift package "${ABSLAYER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" clean' \
    "$BUILD_SCRIPT" | cut -d: -f1
)"
cache_write_line="$(
  grep -nF 'mv -f "$CACHE_PROFILE_TEMP" "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"
build_line="$(
  grep -nF 'swift build "${SWIFT_BUILD_ARGS[@]}"' "$BUILD_SCRIPT" \
    | head -n 1 | cut -d: -f1
)"
verification_line="$(
  grep -nF 'if [[ ! -x "$BIN_DIR/$executable" ]]' "$BUILD_SCRIPT" | cut -d: -f1
)"
success_write_line="$(
  grep -nF 'mv -f "$SUCCESS_PROFILE_TEMP" "$PROFILE_MARKER"' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"

for line in \
  "$success_invalidate_line" \
  "$clean_line" \
  "$cache_write_line" \
  "$build_line" \
  "$verification_line" \
  "$success_write_line"
do
  [[ "$line" =~ ^[0-9]+$ ]]
done

# The old success marker is invalidated before the scratch tree can be cleaned
# or compiled. The resumable cache marker precedes compilation, while success
# remains absent until every release executable has passed verification.
(( success_invalidate_line < clean_line ))
(( clean_line < cache_write_line ))
(( cache_write_line < build_line ))
(( build_line < verification_line ))
(( verification_line < success_write_line ))

# Both marker publications must be atomic same-directory temp-file renames.
grep -Fq 'CACHE_PROFILE_TEMP="$CACHE_PROFILE_MARKER.tmp.$$"' "$BUILD_SCRIPT"
grep -Fq 'printf '\''%s\n'\'' "$BUILD_PROFILE" > "$CACHE_PROFILE_TEMP"' "$BUILD_SCRIPT"
grep -Fq 'SUCCESS_PROFILE_TEMP="$PROFILE_MARKER.tmp.$$"' "$BUILD_SCRIPT"
grep -Fq 'printf '\''%s\n'\'' "$BUILD_PROFILE" > "$SUCCESS_PROFILE_TEMP"' "$BUILD_SCRIPT"

if grep -Fq '> "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT"; then
  echo "Cache profile marker must not be written directly." >&2
  exit 1
fi
if grep -Fq '> "$PROFILE_MARKER"' "$BUILD_SCRIPT"; then
  echo "Success profile marker must not be written directly." >&2
  exit 1
fi

echo "ABSlayer build cache/success profile separation checks passed"
