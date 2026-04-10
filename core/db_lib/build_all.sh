#!/bin/bash

# Build script for all modules
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building all modules..."
echo ""

echo "=== Building C++ modules ==="
"$SCRIPT_DIR/build_cpp.sh"

echo ""
echo "=== Building Rust modules ==="
"$SCRIPT_DIR/build_rust.sh"

echo ""
echo "All modules built successfully!"
