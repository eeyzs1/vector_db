
# 7VecDB 测试框架

本项目为 7VecDB 提供了完整的测试框架，包括单元测试、正确性验证和边界条件测试。

## 测试框架概览

- **Python 测试**: 使用 pytest 框架
- **C++ 测试**: 使用 GoogleTest (GTest) 框架
- **Rust 测试**: 使用 Rust 内置的测试框架

## Python 测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest test/test_vector_db.py

# 显示详细输出
pytest -v

# 运行特定测试类
pytest test/test_vector_db.py::TestCorrectnessAgainstFAISS

# 运行特定测试方法
pytest test/test_vector_db.py::TestBoundaryConditions::test_single_vector
```

### 测试分类

- **单元测试**: `TestVectorIndexCreation`, `TestBasicOperations`
- **正确性测试**: `TestCorrectnessAgainstFAISS` (与 FAISS 结果对比)
- **边界条件测试**: `TestBoundaryConditions`
- **异步测试**: `TestAsyncOperations`

## C++ 测试

### 构建和运行测试

```bash
cd cpp

# 创建构建目录
mkdir -p build && cd build

# 配置 CMake
cmake ..

# 编译
make

# 运行测试
ctest

# 或者直接运行测试可执行文件
./test_flat_l2
```

### 测试说明

- 测试文件位于 `cpp/tests/` 目录
- 当前包含 Flat L2 索引的基本测试

## Rust 测试

### 运行测试

```bash
cd rust

# 运行所有测试
cargo test

# 运行特定 crate 的测试
cargo test -p flat

# 显示详细输出
cargo test -- --nocapture
```

### 测试说明

- 测试代码位于各 crate 的 `lib.rs` 文件底部的 `#[cfg(test)]` 模块中
- 当前为 flat 模块添加了测试

## 测试覆盖率目标

- ✅ 单元测试框架已搭建
- ✅ 正确性验证（与 FAISS 对比）已实现
- ✅ 边界条件测试已添加
- 🔄 更多索引类型的测试待添加

## 扩展测试

要为其他索引类型添加测试，请遵循相应语言的测试模式：

### Python
在 `test/test_vector_db.py` 中添加新的测试类或方法。

### C++
在 `cpp/tests/` 中创建新的测试文件，并在 `CMakeLists.txt` 中添加相应的配置。

### Rust
在对应 crate 的 `lib.rs` 中添加 `#[cfg(test)]` 模块和测试方法。

