#!/bin/bash

# Build script for Rust modules
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust"
PYTHON_DIR="$SCRIPT_DIR/python/vectordb/rust"

mkdir -p "$PYTHON_DIR"
cd "$RUST_DIR"

echo "Building Rust modules..."
cargo build --release

# Copy the compiled shared libraries to Python directory
echo "Copying Rust modules to Python directory..."

# Copy flat
if [ -f "target/release/libflat.so" ]; then
    cp "target/release/libflat.so" "$PYTHON_DIR/_flat.so"
    echo "Copied flat to $PYTHON_DIR/_flat.so"
fi

# Copy flat_ip
if [ -f "target/release/libflat_ip.so" ]; then
    cp "target/release/libflat_ip.so" "$PYTHON_DIR/_flat_ip.so"
    echo "Copied flat_ip to $PYTHON_DIR/_flat_ip.so"
fi

# Copy ivf
if [ -f "target/release/libivf.so" ]; then
    cp "target/release/libivf.so" "$PYTHON_DIR/_ivf.so"
    echo "Copied ivf to $PYTHON_DIR/_ivf.so"
fi

# Copy hnsw
if [ -f "target/release/libhnsw.so" ]; then
    cp "target/release/libhnsw.so" "$PYTHON_DIR/_hnsw.so"
    echo "Copied hnsw to $PYTHON_DIR/_hnsw.so"
fi

# Copy pq
if [ -f "target/release/libpq.so" ]; then
    cp "target/release/libpq.so" "$PYTHON_DIR/_pq.so"
    echo "Copied pq to $PYTHON_DIR/_pq.so"
fi

# Copy lsh
if [ -f "target/release/liblsh.so" ]; then
    cp "target/release/liblsh.so" "$PYTHON_DIR/_lsh.so"
    echo "Copied lsh to $PYTHON_DIR/_lsh.so"
fi

# Copy kdtree
if [ -f "target/release/libkdtree.so" ]; then
    cp "target/release/libkdtree.so" "$PYTHON_DIR/_kdtree.so"
    echo "Copied kdtree to $PYTHON_DIR/_kdtree.so"
fi

# Copy balltree
if [ -f "target/release/libballtree.so" ]; then
    cp "target/release/libballtree.so" "$PYTHON_DIR/_balltree.so"
    echo "Copied balltree to $PYTHON_DIR/_balltree.so"
fi

# Copy annoy
if [ -f "target/release/libannoy.so" ]; then
    cp "target/release/libannoy.so" "$PYTHON_DIR/_annoy.so"
    echo "Copied annoy to $PYTHON_DIR/_annoy.so"
fi

echo "Rust modules built successfully!"
ls -la "$PYTHON_DIR"
