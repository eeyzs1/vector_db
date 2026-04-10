
# Python 绑定优化说明

## 优化内容总结

### 1. 使用 nanobind 替代 pybind11
- 为所有 9 种算法创建了完整的 nanobind 绑定
- 包括：flat, flat_ip, ivf, hnsw, pq, lsh, kdtree, balltree, annoy
- 相比 pybind11，nanobind 提供更小的二进制文件、更快的编译速度和更低的运行时开销
- 所有绑定文件位于 `cpp_nanobind/` 目录

### 2. 增强的类型注解（Type Hints）
- 所有 Python 接口函数都添加了完整的类型注解
- 使用 `numpy.typing.NDArray` 提供精确的 NumPy 数组类型注解
- 使用 `Enum` 类规范参数类型
- 使用 `typing` 模块提供类型安全

### 3. 完整的 async/await 支持
- 新增 `train_async()` 方法，支持异步训练
- 新增 `add_async()` 方法，支持异步添加向量
- 新增 `search_async()` 方法，支持异步搜索
- 使用 `ThreadPoolExecutor` 和 `asyncio` 实现异步操作
- 保持了上下文管理器协议，支持 `with` 语句自动清理资源

### 4. 默认使用 nanobind
- 默认实现从 pybind11 改为 nanobind
- 添加了 `Implementation.CPP_NANOBIND` 枚举选项
- 向后兼容，仍然支持 pybind11 和 Rust 实现

## 文件结构

```
db_lib/
├── cpp_nanobind/
│   ├── CMakeLists.txt           # 更新的 CMake 构建配置
│   ├── bindings_flat.cpp        # Flat L2 nanobind 绑定
│   ├── bindings_flat_ip.cpp     # Flat IP nanobind 绑定
│   ├── bindings_ivf.cpp         # IVF nanobind 绑定
│   ├── bindings_hnsw.cpp        # HNSW nanobind 绑定
│   ├── bindings_pq.cpp          # PQ nanobind 绑定
│   ├── bindings_lsh.cpp         # LSH nanobind 绑定
│   ├── bindings_kdtree.cpp      # KD-Tree nanobind 绑定
│   ├── bindings_balltree.cpp    # Ball-Tree nanobind 绑定
│   └── bindings_annoy.cpp       # Annoy nanobind 绑定
├── python/vectordb/
│   ├── __init__.py               # 模块导出
│   └── index.py                  # 优化的索引包装层（含类型注解和 async/await）
├── build_nanobind.sh             # nanobind 构建脚本
├── test_nanobind.py              # nanobind 测试代码
└── requirements_nanobind.txt     # 依赖项
```

## 使用说明

### 构建 nanobind 扩展

```bash
cd vector_db/core/db_lib
./build_nanobind.sh
```

### 基本同步接口（默认使用 nanobind）

```python
from vectordb import VectorIndex, IndexType, Implementation, create_index

index = create_index("flat_l2", 128, implementation="cpp_nanobind")
index.add(vectors)
distances, labels = index.search(queries, k=10)
```

### 异步接口

```python
import asyncio
from vectordb import create_index

async def main():
    index = create_index("flat_l2", 128, "cpp_nanobind")
    await index.add_async(vectors)
    distances, labels = await index.search_async(queries, k=10)
    index.close()

asyncio.run(main())
```

### 上下文管理器

```python
from vectordb import create_index

with create_index("flat_l2", 128, "cpp_nanobind") as index:
    index.add(vectors)
    distances, labels = index.search(queries, k=10)
# 自动关闭资源
```

### 异步上下文管理器

```python
import asyncio
from vectordb import create_index

async def main():
    async with create_index("flat_l2", 128, "cpp_nanobind") as index:
        await index.add_async(vectors)
        distances, labels = await index.search_async(queries, k=10)

asyncio.run(main())
```

### 支持的索引类型

- `flat_l2` - 暴力搜索（L2 距离）
- `flat_ip` - 暴力搜索（内积）
- `hnsw` - HNSW 索引
- `ivf` - IVF 倒排文件索引
- `pq` - 乘积量化索引
- `lsh` - 局部敏感哈希
- `kd_tree` - KD 树
- `ball_tree` - Ball 树
- `annoy` - Annoy 索引

## 运行测试

```bash
cd vector_db/core/db_lib
python test_nanobind.py
```

## 性能对比

nanobind 相比 pybind11 的优势：
- **更小的二进制文件**：约减少 30-50% 的体积
- **更快的编译速度**：编译时间可减少 20-40%
- **更低的运行时开销**：函数调用开销更低
- **更好的类型系统**：更严格的类型检查
- **更好的错误信息**：更清晰的编译和运行时错误

注意：nanobind 要求 C++17 或更高版本。

## 向后兼容性

- 仍然支持原有的 pybind11 实现
- 仍然支持 Rust 实现
- 可以通过 `implementation` 参数选择实现类型：
  - `"cpp_nanobind"` - 默认，使用 nanobind
  - `"cpp"` - 使用 pybind11
  - `"rust"` - 使用 Rust

