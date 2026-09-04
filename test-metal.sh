#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Build the application metallib, build/discover the test bundle without
# executing numerical tests, then colocate the metallib and run that bundle.
"$ROOT/build-metal.sh"
swift test --disable-sandbox list >/dev/null
BIN_DIR="$(swift build --show-bin-path)"
while IFS= read -r -d '' test_bin; do
  cp "$BIN_DIR/mlx.metallib" "$test_bin/mlx.metallib"
done < <(find "$BIN_DIR" -type d -path '*PackageTests.xctest/Contents/MacOS' -print0)
swift test --disable-sandbox --skip-build
