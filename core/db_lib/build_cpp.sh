#!/bin/bash

# Build script for C++ modules
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/cpp"
PYTHON_DIR="$SCRIPT_DIR/python/vectordb/cpp"

mkdir -p "$PYTHON_DIR"
mkdir -p "$CPP_DIR/build"
cd "$CPP_DIR/build"

echo "Building C++ modules..."
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "C++ modules built successfully!"
ls -la "$PYTHON_DIR"
