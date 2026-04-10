
#!/bin/bash

set -e

echo "Building nanobind extensions..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &amp;&amp; pwd)"
cd "$SCRIPT_DIR/cpp_nanobind"

if [ ! -d "build" ]; then
    mkdir -p build
fi

cd build

echo "Running CMake..."
cmake ..

echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Extensions are located in: $SCRIPT_DIR/python/vectordb/cpp/"

