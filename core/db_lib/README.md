# 7VecDB - Modular Vector Database Library

高性能模块化向量数据库库，支持 C++ 和 Rust 实现，提供多种索引算法。

## 架构设计

本项目采用模块化设计，核心思想是：**改一个算法，只重编那个算法，Python 壳和核心都不用全编译。**

### 目录结构

```
db_lib/
├── cpp/                    # C++ 实现
│   ├── core/               # 公共核心库 (vectordb_core)
│   ├── algorithms/         # 独立算法模块
│   │   ├── flat/          # Flat L2 索引
│   │   ├── hnsw/          # HNSW (预留)
│   │   ├── ivf/           # IVF (预留)
│   │   └── ...
│   └── CMakeLists.txt
├── rust/                   # Rust 实现
│   ├── vectordb-core/     # 公共核心库
│   ├── vectordb-flat/     # Flat 索引模块
│   ├── vectordb-hnsw/     # HNSW (预留)
│   └── Cargo.toml
├── python/                 # Python 包装层
│   └── vectordb/
│       ├── __init__.py
│       ├── index.py        # 统一入口
│       ├── cpp/            # C++ 编译的扩展
│       └── rust/           # Rust 编译的扩展
├── build_cpp.sh            # C++ 编译脚本
├── build_rust.sh           # Rust 编译脚本
└── build_all.sh            # 全部编译脚本
```

## 编译

### 编译 C++ 模块

```bash
./build_cpp.sh
```

### 编译 Rust 模块

```bash
./build_rust.sh
```

### 编译所有模块

```bash
./build_all.sh
```

## 使用方法

### 基本同步接口

```python
import numpy as np
from vectordb import VectorIndex, IndexType, Implementation

# 创建 Flat L2 索引 (C++ 实现)
index = VectorIndex(
    index_type=IndexType.FLAT_L2,
    dimension=128,
    implementation=Implementation.CPP
)

# 或者使用便捷函数
from vectordb import create_index
index = create_index("flat_l2", 128, "cpp")

# 添加向量
vectors = np.random.rand(10000, 128).astype(np.float32)
index.add(vectors)

# 搜索
queries = np.random.rand(10, 128).astype(np.float32)
distances, labels = index.search(queries, k=5)

print(f"Index size: {index.size()}")
print(f"Distances shape: {distances.shape}")
print(f"Labels shape: {labels.shape}")
```

### 异步接口 (Async/Await)

```python
import asyncio
import numpy as np
from vectordb import create_index

async def main():
    index = create_index("flat_l2", 128, "cpp")
    
    # 添加向量
    vectors = np.random.rand(10000, 128).astype(np.float32)
    index.add(vectors)
    
    # 异步搜索
    queries = np.random.rand(10, 128).astype(np.float32)
    distances, labels = await index.search_async(queries, k=5)
    
    print(f"Distances shape: {distances.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 关闭索引，释放线程池资源
    index.close()

asyncio.run(main())
```

### 上下文管理器 (With 语句)

```python
import numpy as np
from vectordb import create_index

with create_index("flat_l2", 128, "cpp") as index:
    vectors = np.random.rand(10000, 128).astype(np.float32)
    index.add(vectors)
    
    queries = np.random.rand(10, 128).astype(np.float32)
    distances, labels = index.search(queries, k=5)
    
    print(f"Results ready!")
# 自动关闭，无需手动调用 close()
```

### 使用不同索引类型

```python
import numpy as np
from vectordb import create_index

# Flat IP (内积) 索引
index_ip = create_index("flat_ip", 128, "cpp")

# IVF 索引 (需要训练)
index_ivf = create_index("ivf", 128, "cpp", nlist=100)
training_data = np.random.rand(10000, 128).astype(np.float32)
index_ivf.train(training_data)
index_ivf.add(vectors)
index_ivf.set_nprobe(10)  # 设置搜索时检查的簇数量

# HNSW 索引
index_hnsw = create_index("hnsw", 128, "cpp", M=16, ef_construction=200)
index_hnsw.set_ef_search(100)  # 设置搜索时的候选列表大小
```

### 使用 Rust 实现

```python
import numpy as np
from vectordb import create_index

# 创建 Rust 实现的 Flat L2 索引
index = create_index("flat_l2", 128, "rust")

# 使用方式与 C++ 实现完全一致
vectors = np.random.rand(10000, 128).astype(np.float32)
index.add(vectors)

queries = np.random.rand(10, 128).astype(np.float32)
distances, labels = index.search(queries, k=5)
```

## 支持的索引类型

### C++ 实现
- ✅ **FLAT_L2**: 暴力搜索 L2 距离
- ✅ **FLAT_IP**: 暴力搜索内积
- ✅ **HNSW**: Hierarchical Navigable Small Worlds
- ✅ **IVF**: Inverted File
- ✅ **PQ**: Product Quantization
- ✅ **LSH**: Locality-Sensitive Hashing
- ✅ **KD_TREE**: K-Dimensional Tree
- ✅ **BALL_TREE**: Ball Tree
- ✅ **ANNOY**: Approximate Nearest Neighbors Oh Yeah

### Rust 实现
- ✅ **FLAT_L2**: 暴力搜索 L2 距离
- ✅ **FLAT_IP**: 暴力搜索内积
- ⏳ **HNSW**: Hierarchical Navigable Small Worlds (待实现)
- ⏳ **IVF**: Inverted File (待实现)
- ⏳ **PQ**: Product Quantization (待实现)
- ⏳ **LSH**: Locality-Sensitive Hashing (待实现)
- ⏳ **KD_TREE**: K-Dimensional Tree (待实现)
- ⏳ **BALL_TREE**: Ball Tree (待实现)
- ⏳ **ANNOY**: Approximate Nearest Neighbors Oh Yeah (待实现)

## 模块化优势

1. **增量编译**: 只重新编译修改过的算法模块
2. **快速开发**: 核心库稳定后，算法模块可独立开发测试
3. **灵活替换**: 可以轻松替换不同实现的同一算法
4. **统一接口**: 所有算法通过同一个 Python 接口访问

## 开发新算法

### C++ 算法

1. 在 `cpp/algorithms/` 下创建新目录
2. 实现算法，链接 `vectordb_core`
3. 在 `CMakeLists.txt` 中添加新模块
4. 运行 `./build_cpp.sh` 编译

### Rust 算法

1. 在 `rust/` 下创建新 crate
2. 依赖 `vectordb-core`
3. 在工作区 `Cargo.toml` 中添加新成员
4. 运行 `./build_rust.sh` 编译

## 性能优化

- 核心库包含 SIMD 优化的距离计算
  - AVX2 支持（默认）
  - AVX-512 支持（硬件兼容时自动启用）
- 多线程支持
- 内存布局优化
- OpenBLAS 集成（C++）
- ndarray 库支持（Rust）

### SIMD 优化说明

#### Rust 实现

Rust 核心库包含多层次的 SIMD 优化：

- **AVX-512F**: 当硬件支持时自动启用，提供最高性能
- **AVX2**: 为不支持 AVX-512 的系统提供优化
- **标量**: 兼容性回退

Rust 代码通过条件编译在编译时选择最优实现。

#### C++ 实现

C++ 核心库同样包含多层次 SIMD 支持：

- **AVX-512F** 优化的距离计算
- **AVX2** 优化的距离计算
- **OpenBLAS** 集成，用于批量矩阵运算

### 编译配置

#### C++ OpenBLAS 支持

C++ 构建默认启用 OpenBLAS 支持。如果系统上安装了 OpenBLAS，它将被自动检测和链接。

要禁用 OpenBLAS：

```bash
cd cpp
mkdir -p build && cd build
cmake -DUSE_OPENBLAS=OFF ..
make
```

#### Rust 特性标志

Rust 实现支持以下特性标志：

- `simd` (默认): 启用 SIMD 优化
- `avx512`: 明确启用 AVX-512 优化（需要硬件支持）

要在编译时启用 AVX-512 优化：

```bash
RUSTFLAGS="-C target-feature=+avx512f" ./build_rust.sh
```

或者修改 Cargo.toml 中的特性配置。

### ndarray 库（Rust）

Rust 实现现在包含 `ndarray` 库依赖，用于简化数组操作。ndarray 提供：

- 高效的多维数组操作
- 与 NumPy 类似的 API
- 优化的内存布局

所有 Rust 模块现在都可以使用 ndarray 进行更高效的数组操作。

## API 文档

### 生成 C++ 文档 (Doxygen)

```bash
cd cpp
doxygen Doxyfile
# 文档将生成在 cpp/docs/html/ 目录下
```

### 生成 Rust 文档 (rustdoc)

```bash
cd rust
cargo doc --no-deps --open
```

### Python API

完整的 Python API 类型注解已内联在 `python/vectordb/index.py` 中，支持 IDE 自动补全和类型检查。

主要类和函数：
- `VectorIndex`: 向量索引的主类
- `IndexType`: 枚举类型，指定索引类型
- `Implementation`: 枚举类型，指定实现语言 (CPP/RUST)
- `create_index()`: 便捷函数，用于创建索引

主要方法：
- `VectorIndex.add(vectors)`: 添加向量到索引
- `VectorIndex.search(queries, k)`: 同步搜索最近邻
- `VectorIndex.search_async(queries, k)`: 异步搜索最近邻
- `VectorIndex.train(vectors)`: 训练索引 (需要训练的算法如 IVF)
- `VectorIndex.size()`: 获取索引中的向量数量
- `VectorIndex.get_dimension()`: 获取向量维度
- `VectorIndex.close()`: 关闭索引，释放资源

## Python 绑定优化

本项目提供了优化的 Python 绑定，详见 `PYTHON_BINDING_OPTIMIZATION.md`。

主要优化包括：
- 完整的类型注解 (Type Hints)
- Async/Await 异步接口支持
- 上下文管理器 (With 语句) 支持
- Nanobind 绑定示例 (用于性能对比)

## 测试

本项目提供了完整的测试框架，包括单元测试、正确性验证和边界条件测试。

详细的测试说明请参见 `TESTING_README.md`。

### 快速开始测试

```bash
# Python 测试
pip install -r requirements.txt
pytest

# C++ 测试
cd cpp && mkdir -p build && cd build
cmake .. && make && ctest

# Rust 测试
cd rust
cargo test
```

### 测试框架

- **Python**: pytest + FAISS 对比验证
- **C++**: GoogleTest (GTest)
- **Rust**: 内置测试框架

测试覆盖范围：
- ✅ 单元测试
- ✅ 正确性验证（与 FAISS 结果对比）
- ✅ 边界条件测试

## 代码组织改进

### VectorStorage 类完善

完善了 `VectorStorage` 类中的 transposed 相关功能，现在该类包含：

- **`transpose()`**：私有方法，实现向量矩阵的转置操作
- **`transposed_data()`**：公共方法，获取转置后的数据（支持按需延迟转置）
- **`is_transposed()`**：公共方法，检查数据是否已经转置
- **`clear_transposed()`**：公共方法，清除转置缓存
- **改进的 `add()` 方法**：添加新向量时会自动清除转置缓存

这些改进使得转置存储功能完整可用，提高了某些搜索场景下的性能。

### 新增接口定义

添加了 `IndexInterface` 抽象基类，为所有向量索引提供统一的接口契约：

```cpp
class IndexInterface {
public:
    virtual ~IndexInterface() = default;
    
    virtual void add(size_t n, const float* x) = 0;
    virtual void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const = 0;
    
    virtual void train(size_t n, const float* x) {}
    virtual size_t get_ntotal() const = 0;
    virtual size_t get_dimension() const = 0;
};
```

该接口确保所有索引实现遵循一致的 API，提高了代码的可维护性和可扩展性。

### 架构优势

1. **统一接口**：所有索引算法通过 `IndexInterface` 提供一致的 API
2. **完整功能**：VectorStorage 的 transposed 功能完整实现
3. **延迟转置**：按需计算转置，节省内存和计算资源
4. **自动缓存失效**：添加新向量时自动清除转置缓存，确保数据一致性

