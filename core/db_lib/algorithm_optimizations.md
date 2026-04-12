# C++ 向量数据库算法优化总结

本文档对项目中各向量搜索算法的C++实现所采用的优化手段进行系统性总结，按照算法原理、调优方法及其原理三个维度组织。

---

## 目录

1. [通用基础设施优化](#1-通用基础设施优化)
2. [FlatL2 — 暴力L2搜索](#2-flatl2--暴力l2搜索)
3. [FlatIP — 暴力内积搜索](#3-flatip--暴力内积搜索)
4. [HNSW — 层级可导航小世界图](#4-hnsw--层级可导航小世界图)
5. [IVF — 倒排文件索引](#5-ivf--倒排文件索引)
6. [PQ — 乘积量化](#6-pq--乘积量化)
7. [LSH — 局部敏感哈希](#7-lsh--局部敏感哈希)

---

## 1. 通用基础设施优化

> 源文件：[vectordb_core.h](cpp/core/vectordb_core.h)

所有算法共享的基础设施层优化，是各算法性能提升的底层支撑。

### 1.1 SIMD向量化距离计算

| 距离类型 | AVX-512实现 | AVX2实现 | 标量回退 |
|---------|------------|---------|---------|
| L2距离 | `_mm512_fmadd_ps` 处理16个float | `_mm256_fmadd_ps` 处理8个float | 逐元素循环 |
| 内积距离 | `_mm512_fmadd_ps` 直接乘加 | `_mm256_fmadd_ps` 直接乘加 | 逐元素循环 |

**原理**：SIMD（Single Instruction Multiple Data）允许一条指令同时处理多个数据元素。AVX-512的512位寄存器可容纳16个32位浮点数，AVX2的256位寄存器可容纳8个，相比标量循环可获得8~16倍的吞吐提升。FMA（Fused Multiply-Add）指令将乘法和加法合并为一条指令，进一步减少指令数并提高浮点精度（仅一次舍入）。

### 1.2 内存对齐分配器

```cpp
template <typename T, size_t Alignment>
class AlignedAllocator { ... };
```

**原理**：SIMD指令的对齐加载（如`_mm512_load_ps`）要求数据地址按64字节对齐，非对齐加载（`_mm512_loadu_ps`）虽可工作但性能略低。通过自定义对齐分配器，确保关键数据结构（如LSH的哈希函数矩阵）在分配时就满足对齐要求，可使用更高效的对齐加载指令。

### 1.3 OpenBLAS批量距离计算接口

```cpp
#ifdef USE_OPENBLAS
inline void compute_batch_distances_l2(...) { cblas_sdot(...); }
inline void compute_batch_distances_ip(...) { cblas_sgemm(...); }
#endif
```

**原理**：OpenBLAS是高度优化的BLAS实现，其矩阵乘法（SGEMM）和向量点积（SDOT）针对特定CPU微架构做了汇编级优化，包括缓存分块、寄存器重排、指令流水线等。对于大批量距离计算，调用OpenBLAS可超越手写SIMD代码的性能。

---

## 2. FlatL2 — 暴力L2搜索

> 源文件：[flat.h](cpp/algorithms/flat/flat.h), [flat.cpp](cpp/algorithms/flat/flat.cpp)

### 算法原理

暴力搜索（Brute-force）是最直接的近邻搜索方法：对每个查询向量，计算其与数据库中所有向量的L2距离，选出距离最小的k个。时间复杂度O(n·d)，其中n为向量数，d为维度。虽然复杂度高，但精度为100%，常作为其他算法的精度基准。

### 调优方法

#### 2.1 转置数据布局 + AVX2批量距离计算

**方法**：将原始行优先数据（`[n, d]`）转置为列优先布局（`[d, n]`），利用`compute_batch_distances_transposed`同时计算8个向量与查询的距离。

**原理**：

- **缓存友好性**：行优先布局中，8个向量的同一维度数据在内存中不相邻（间隔d个float），导致缓存未命中。转置后，8个向量的同一维度数据连续存储，一次缓存行加载（64字节=16个float）即可服务多个向量。
- **SIMD利用率**：转置布局下，查询向量的每个维度被广播（`_mm256_set1_ps`），8个数据库向量的对应维度被连续加载（`_mm256_loadu_ps`），实现一次SIMD指令同时计算8个向量的部分距离。
- **8维展开**：内层循环每次处理8个维度（8次`_mm256_set1_ps` + `_mm256_loadu_ps` + `_mm256_fmadd_ps`），减少循环判断开销，提高指令级并行度。

#### 2.2 软件预取

**方法**：在距离计算循环中插入`__builtin_prefetch`指令，预取后续数据块。

```cpp
__builtin_prefetch(base_ptr + (dim_idx + 8) * ntotal, 0, 3);
__builtin_prefetch(base_ptr + (dim_idx + 16) * ntotal, 0, 3);
```

**原理**：CPU缓存未命中会导致数百个时钟周期的停顿。软件预取通过提前发出数据加载请求，使数据在真正使用前已到达L1/L2缓存，隐藏内存访问延迟。参数`0`表示只读，`3`表示高时间局部性（尽量保留在缓存中）。

#### 2.3 标准布局下的8路循环展开

**方法**：当未使用转置布局时，每次迭代同时计算8个向量的距离，并手动展开top-k插入逻辑。

**原理**：
- **减少分支预测失败**：8个距离计算结果一次性获得，减少了循环条件判断的次数。
- **指令级并行**：8个独立的距离计算之间无数据依赖，CPU可以乱序执行。
- **预取配合**：同时预取后续8个和16个向量的数据。

#### 2.4 二分查找Top-k维护

**方法**：维护一个大小为k的有序数组，新距离通过二分查找确定插入位置，用`memmove`移动后续元素。

**原理**：
- **时间复杂度**：二分查找O(log k) + 移动O(k)，对于k通常≤1024的场景，比维护完全排序的优先队列更高效。
- **缓存友好**：top-k数组大小有限（k个float + k个size_t），完全驻留在L1缓存中。
- **提前剪枝**：只有当新距离小于当前第k大距离时才执行插入，大部分向量被跳过。

#### 2.5 多线程并行搜索

**方法**：
- **单查询**：`search_parallel`将数据库向量均匀分配到多个线程，每个线程维护局部top-k，最后合并。
- **多查询**：将查询向量分配到不同线程，每个线程独立执行`search_single`。

**原理**：
- **数据并行**：暴力搜索的各向量距离计算相互独立，天然适合并行化。
- **自适应线程数**：根据向量总数和维度动态调整线程数——高维度时减少线程数以避免内存带宽饱和，大数据集时增加线程数以充分利用多核。
- **最小粒度控制**：每个线程至少处理1000个向量，避免线程管理开销超过计算收益。

#### 2.6 自适应块大小

**方法**：转置布局下，根据数据规模选择不同的块大小——小数据集（<20万）使用16384，大数据集使用4096。

**原理**：小数据集可全部放入L3缓存，使用大块减少循环开销；大数据集受限于缓存容量，使用小块确保工作集适配缓存，减少缓存驱逐。

---

## 3. FlatIP — 暴力内积搜索

> 源文件：[flat_ip.h](cpp/algorithms/flat_ip/flat_ip.h), [flat_ip.cpp](cpp/algorithms/flat_ip/flat_ip.cpp)

### 算法原理

与FlatL2类似，但距离度量改为内积（Inner Product）。搜索目标是找出与查询内积最大的k个向量。内积距离`<q, v> = Σ(q_i × v_i)`，无需先做减法再平方，计算更简单。

### 调优方法

FlatIP的优化与FlatL2高度相似，以下仅列出关键差异：

#### 3.1 内积专用的转置批量计算

**方法**：转置布局下的AVX2批量计算直接使用`_mm256_fmadd_ps(q, v, dist_vec)`，无需先做减法。

**原理**：L2距离需要`diff = q - v`再`diff * diff`，而内积只需`q * v`直接累加，少一次减法操作，每个维度节省一条SIMD指令。

#### 3.2 最大堆Top-k（取最大内积）

**方法**：top-k数组初始化为`-float_max`，插入条件为`dist > top_distances[k-1]`，二分查找方向与L2相反。

**原理**：L2距离越小越相似（取最小），内积越大越相似（取最大），排序方向相反但维护逻辑对称。

#### 3.3 提取`insert_into_top_k`辅助函数

**方法**：将top-k插入逻辑封装为内联函数，避免8路展开时的代码重复。

**原理**：代码复用减少维护成本，编译器内联后无函数调用开销。

---

## 4. HNSW — 层级可导航小世界图

> 源文件：[hnsw.h](cpp/algorithms/hnsw/hnsw.h), [hnsw.cpp](cpp/algorithms/hnsw/hnsw.cpp)

### 算法原理

HNSW（Hierarchical Navigable Small World）是一种基于多层图结构的近似最近邻搜索算法：

- **层级结构**：每个节点被分配一个随机层级（指数分布），高层节点稀疏，低层节点密集。
- **插入**：从最高层入口点开始贪心搜索到节点所在层级，逐层向下建立邻居连接。
- **搜索**：从最高层贪心下降到第0层，在第0层执行宽度优先搜索（ef_search宽度），返回最近的k个结果。

核心参数：M（每层最大邻居数）、ef_construction（构建时搜索宽度）、ef_search（查询时搜索宽度）。

### 调优方法

#### 4.1 紧凑邻居存储

**方法**：查询时使用`neighbors_compact_`（`uint32_t`数组）和`neighbor_counts_`（`uint16_t`数组）替代原始的`vector<vector<int32_t>>`。

```
原始存储：vector<vector<int32_t>> neighbors_  — 每个节点一个vector，堆分配
紧凑存储：vector<uint32_t> neighbors_compact_  — 连续数组，固定slot大小
         vector<uint16_t> neighbor_counts_      — 每层邻居计数
```

**原理**：
- **内存节省**：`uint32_t`比`int32_t`节省的虽不多（4字节→4字节），但`uint16_t`计数器比vector的size字段更紧凑。更重要的是，固定slot布局消除了vector的3指针开销（24字节/节点）。
- **缓存友好**：连续数组布局使邻居数据在内存中紧密排列，一次缓存行加载可获取多个节点的邻居信息。原始vector的堆分配导致数据分散在堆中，缓存命中率低。
- **层级偏移表**：`compact_level_offsets_[32]`预计算每层在slot中的偏移，O(1)定位任意节点的某层邻居。

#### 4.2 预计算向量范数

**方法**：`precompute_norms()`在搜索前计算所有向量的L2范数平方。

**原理**：L2距离可分解为`||q-v||² = ||q||² + ||v||² - 2<q,v>`。预计算`||v||²`后，距离计算可转化为一次内积加两次查表，为后续BLAS优化提供基础。

#### 4.3 批量邻居距离计算

**方法**：在`search_layer_impl`中，先收集所有未访问的邻居ID到`batch_ids`数组，再批量计算距离。

```cpp
uint32_t* batch_ids = static_cast<uint32_t*>(alloca((2 * M_) * sizeof(uint32_t)));
float* batch_dists = static_cast<float*>(alloca((2 * M_) * sizeof(float)));
// 收集未访问邻居
for (uint16_t ni = 0; ni < nb_count; ++ni) { ... batch_ids[n_new++] = neighbor; }
// 批量计算距离
for (size_t i = 0; i < n_new; ++i) {
    batch_dists[i] = distance::compute_l2_distance(query, base + batch_ids[i] * dim, dim);
}
```

**原理**：
- **缓存局部性**：先过滤已访问节点，再对剩余节点顺序访问其向量数据。相比边遍历邻居边计算距离，批量处理使向量数据的访问模式更规则，预取器更有效。
- **为BLAS集成预留接口**：批量ID数组可直接传递给BLAS的批量距离计算函数。

#### 4.4 自定义双堆搜索

**方法**：使用手写的最大堆（候选集）和最小堆（结果集），替代`std::priority_queue`。

```cpp
// 候选集：最大堆，优先处理距离最近的候选
auto sift_up_max = [&](size_t idx) { ... };
auto sift_down_max = [&](size_t idx) { ... };
// 结果集：最小堆，堆顶为最远结果，便于剪枝
auto sift_up_min = [&](size_t idx) { ... };
auto sift_down_min = [&](size_t idx) { ... };
```

**原理**：
- **消除抽象开销**：`std::priority_queue`的函数调用和适配器层引入额外开销，手写堆操作直接操作裸数组，编译器可完全内联。
- **双堆协同剪枝**：候选集最大堆堆顶为最远候选，结果集最小堆堆顶为最远结果。当最远候选比最远结果还远时，搜索可提前终止。
- **栈分配**：使用`alloca`在栈上分配堆数据，避免堆分配的开销和碎片。

#### 4.5 提前终止策略

**方法**：当候选集中最近候选比结果集中最远结果还远，且结果集已满时，终止搜索。

```cpp
if (cand_dists[0] > res_dists[0] && res_size >= ef) { break; }
```

**原理**：HNSW搜索的核心效率来自贪心剪枝。当所有剩余候选都不可能产生更近的结果时，继续搜索是浪费的。此条件在ef较大时可显著减少不必要的距离计算。

#### 4.6 递增访问标记

**方法**：使用递增的`visit_mark`替代清零visited数组。

```cpp
state.visit_mark++;
if (state.visit_mark == 0) {
    state.visited.assign(ntotal, 0);  // 溢出时才重置
    state.visit_mark = 1;
}
visited[neighbor] = visit_mark;  // 标记访问
if (visited[neighbor] == visit_mark) continue;  // 检查已访问
```

**原理**：传统方法每次查询前需O(n)清零visited数组。递增标记法将清零操作从O(n)降为O(1)（仅递增标记值），只有当int32_t溢出时才执行一次O(n)重置。对于百万级数据集，每次查询节省数毫秒。

#### 4.7 插入排序整理最终结果

**方法**：搜索结束后，对结果数组使用插入排序。

```cpp
for (size_t i = 1; i < count; ++i) {
    float key_d = out_dists[i];
    int32_t key_id = out_ids[i];
    size_t j = i;
    while (j > 0 && out_dists[j - 1] > key_d) { ... }
}
```

**原理**：双堆搜索产出的结果集近似有序（最小堆的数组表示），插入排序在近乎有序的数据上接近O(n)，比快速排序的O(n log n)更快。

#### 4.8 插入与查询的分离路径

**方法**：
- `search_layer_impl`：使用紧凑邻居存储，用于查询阶段
- `search_layer_impl_no_blas`：使用原始vector存储，用于插入阶段

**原理**：插入阶段需要动态修改邻居列表（添加/替换邻居），vector的动态大小特性更合适。查询阶段只读访问，紧凑连续数组的缓存性能更优。分离路径避免了在紧凑存储上支持修改操作的复杂性。

---

## 5. IVF — 倒排文件索引

> 源文件：[ivf.h](cpp/algorithms/ivf/ivf.h), [ivf.cpp](cpp/algorithms/ivf/ivf.cpp)

### 算法原理

IVF（Inverted File Index）将向量空间通过K-means聚类划分为nlist个区域（倒排列表）。搜索时，只在与查询最近的nprobe个聚类中查找，将搜索范围从全库缩小到部分区域，实现亚线性搜索时间。

### 调优方法

#### 5.1 K-Means++初始化

**方法**：使用K-Means++算法选择初始聚类中心——每个新中心以正比于到最近中心距离的概率被选中。

**原理**：随机初始化可能导致中心聚集，需要更多迭代收敛且效果差。K-Means++保证初始中心分布均匀，以O(log k)的期望近似比接近最优聚类，减少迭代次数并提升聚类质量。

#### 5.2 K-Means提前终止

**方法**：当质心最大偏移量小于阈值（1e-6）时终止迭代。

**原理**：聚类后期质心移动极小，继续迭代几乎不改善结果但浪费计算。提前终止可将迭代次数从固定的max_iter减少到实际需要的次数（通常5~15次）。

#### 5.3 空聚类处理

**方法**：当某聚类变空时，从最大聚类中分裂——复制最大聚类的质心并添加随机扰动。

**原理**：空聚类导致倒排列表浪费，且减少有效搜索区域。分裂最大聚类既保持了聚类数，又将过大聚类细分，提升搜索精度。

#### 5.4 连续聚类布局

**方法**：`build_cluster_layout()`将各聚类的向量重新排列到连续内存中。

```
cluster_vectors_:     [cluster0_vecs | cluster1_vecs | ... | clusterN_vecs]
cluster_vector_offsets_: [0, offset1, offset2, ...]
cluster_vector_sizes_:   [size0, size1, size2, ...]
cluster_original_ids_:   [id0_0, id0_1, ... | id1_0, id1_1, ... | ...]
cluster_vector_norms_:   [norm0_0, norm0_1, ... | norm1_0, norm1_1, ... | ...]
```

**原理**：
- **缓存友好**：搜索时需要顺序扫描某聚类的所有向量，连续布局使向量数据在内存中紧密排列，一次缓存行加载可获取多个向量数据。原始的倒排列表存储（`vector<vector<size_t>>`）中，向量数据通过间接索引访问基类存储，导致跳跃式内存访问。
- **向量化友好**：连续布局可直接传递给BLAS的矩阵向量乘法（SGEMV），实现批量距离计算。

#### 5.5 预计算聚类向量范数

**方法**：`cluster_vector_norms_`存储每个向量的L2范数平方。

**原理**：L2距离分解`||q-v||² = ||q||² + ||v||² - 2<q,v>`。预计算`||v||²`后，距离计算转化为一次内积加两次查表。内积部分可用BLAS的SGEMV高效计算，范数查表为O(1)。

#### 5.6 nth_element选择最近聚类

**方法**：使用`std::nth_element`而非完全排序来选择最近的nprobe个聚类。

```cpp
std::nth_element(cluster_dists.begin(),
                 cluster_dists.begin() + nprobe,
                 cluster_dists.end());
```

**原理**：`nth_element`只需O(n)时间找到前nprobe小的元素，而完全排序需要O(n log n)。nlist通常为数百到数千，此优化可节省显著时间。

#### 5.7 OpenMP并行搜索

**方法**：使用`#pragma omp parallel for schedule(dynamic, 1)`并行处理多个查询。

**原理**：
- **查询间并行**：不同查询的搜索相互独立，天然适合并行。
- **动态调度**：不同查询的搜索时间可能差异很大（不同聚类的向量数不同），动态调度实现负载均衡。
- **线程局部数据**：每个线程维护独立的`cluster_dists`和`heap`，避免锁竞争。

#### 5.8 最大堆Top-k + 距离截断

**方法**：在每个聚类的搜索中使用最大堆维护top-k，堆顶为最远结果，作为截断阈值。

```cpp
float cutoff = (heap.size() >= k) ? heap.front().first : std::numeric_limits<float>::max();
if (dist < cutoff) { ... }
```

**原理**：当堆已满时，只有距离小于当前第k远的向量才需要插入。对于大部分向量，一次比较即可跳过，避免不必要的堆操作。

---

## 6. PQ — 乘积量化

> 源文件：[pq.h](cpp/algorithms/pq/pq.h), [pq.cpp](cpp/algorithms/pq/pq.cpp)

### 算法原理

乘积量化（Product Quantization）将高维向量切分为M个子向量，每个子空间独立进行K-means聚类（产生ksub=2^nbits个质心）。原始向量被编码为M个质心索引（每个1字节），实现大幅压缩。搜索时采用距离表查找法：先构建M×ksub的距离查找表，再通过查表累加得到近似距离。

### 调优方法

#### 6.1 SIMD加速的K-means训练

**方法**：对常见子维度（dim_sub=8, 16）使用AVX2/AVX512硬编码的K-means迭代。

```cpp
if (dim_sub == 8) {
    __m256 v = _mm256_loadu_ps(vec);
    for (size_t k = 0; k < ksub; ++k) {
        __m256 c = _mm256_loadu_ps(centroids + k * 8);
        __m256 diff = _mm256_sub_ps(v, c);
        __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
        float dist = hsum_avx2(sq);
        ...
    }
}
```

**原理**：dim_sub=8恰好填满一个AVX2寄存器，dim_sub=16恰好填满两个AVX2寄存器，无需处理尾部元素，SIMD利用率100%。训练阶段的距离计算和质心累加均被向量化，加速比可达4~8倍。

#### 6.2 转置码本 + 基于内积的快速编码

**方法**：
- 预计算转置码本`codebooks_t_`：将`[ksub, dim_sub]`布局转为`[dim_sub, ksub]`
- 预计算质心范数`centroid_norms_`
- 编码时利用`score = 2·<q_sub, c> - ||c||²`找最近质心

```cpp
// AVX512: 同时处理16个质心
for (size_t k = 0; k + 15 < ksub_; k += 16) {
    __m512 dot_acc = _mm512_setzero_ps();
    for (size_t j = 0; j < dim_sub; ++j) {
        __m512 vj = _mm512_set1_ps(vec_sub[j]);      // 广播查询维度
        __m512 cj = _mm512_loadu_ps(ct + j * ksub_ + k); // 加载16个质心的该维度
        dot_acc = _mm512_fmadd_ps(vj, cj, dot_acc);  // 累加内积
    }
    __m512 scores = _mm512_sub_ps(
        _mm512_mul_ps(dot_acc, _mm512_set1_ps(2.0f)),
        _mm512_loadu_ps(cn + k));  // score = 2*dot - ||c||²
}
```

**原理**：
- **距离分解**：`||q-c||² = ||q||² + ||c||² - 2<q,c>`。最小化L2距离等价于最大化`2<q,c> - ||c||²`（`||q||²`对所有质心相同可忽略）。
- **转置码本**：原始码本按`[ksub, dim_sub]`存储，计算内积需跳跃访问。转置为`[dim_sub, ksub]`后，16个质心的同一维度连续存储，可一次AVX512加载并广播查询维度，实现16个质心内积的并行计算。
- **避免减法**：直接最大化`2<q,c> - ||c||²`而非最小化`||q-c||²`，省去了逐元素减法和平方操作。

#### 6.3 转置编码 + Gather指令加速搜索

**方法**：将编码数据从行优先`[ntotal, M]`转置为列优先`[M, ntotal]`，搜索时使用AVX512 Gather指令批量查表。

```cpp
// AVX512: 一次处理16个向量
__m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i));
__m512i c32 = _mm512_cvtepu8_epi32(c16);          // 16个uint8转int32
__m512 g = _mm512_i32gather_ps(c32, tab, 4);       // 根据索引从距离表gather
```

**原理**：
- **转置编码**：原始编码按向量存储，查表时每个向量的M个编码分散在M个不同位置。转置后，同一子空间的所有向量编码连续存储，可批量加载16个uint8编码。
- **Gather指令**：`_mm512_i32gather_ps`根据16个索引值从距离表中同时取出16个距离值，一条指令完成16次查表，比标量查表快一个数量级。
- **uint8→int32转换**：`_mm512_cvtepu8_epi32`将16个字节扩展为16个int32，作为Gather的索引。

#### 6.4 专用搜索路径（M=8, M=16）

**方法**：为M=8和M=16提供硬编码的搜索循环，使用8或16个AVX512累加寄存器。

```cpp
// M=16: 使用8对累加器，每对处理2个子空间
__m512 d0, d1, d2, d3, d4, d5, d6, d7;  // 第一组16个向量
__m512 e0, e1, e2, e3, e4, e5, e6, e7;  // 第二组16个向量
for (size_t m = 0; m < 16; ++m) {
    __m512 g = _mm512_i32gather_ps(c32, tab, 4);
    switch (m >> 1) {  // 分配到8个累加器
        case 0: d0 = _mm512_add_ps(d0, g); break;
        ...
    }
}
// 最终归约
d0 = _mm512_add_ps(d0, d1); d2 = _mm512_add_ps(d2, d3); ...
```

**原理**：
- **寄存器充分利用**：AVX-512有32个ZMM寄存器，M=16时使用16个累加寄存器（分两组处理32个向量），消除寄存器溢出。
- **减少归约次数**：每2个子空间共享一个累加器，最终只需8次加法归约而非16次。
- **32向量批处理**：一次处理32个向量（两组各16个），最大化Gather指令的吞吐。

#### 6.5 栈上距离查找表

**方法**：距离查找表分配在栈上而非堆上。

```cpp
float dis_table_buf[16 * 256];  // M=16, ksub=256
float* dis_table = dis_table_buf;
```

**原理**：距离查找表大小固定（M × ksub × 4字节，最大16×256×4=16KB），适合栈分配。栈分配无malloc/free开销，且数据在L1缓存附近，访问延迟极低。

#### 6.6 OpenMP并行搜索

**方法**：使用`#pragma omp parallel for schedule(dynamic, 1)`并行处理查询。

**原理**：与IVF相同，查询间并行+动态调度。每个线程维护独立的距离表和堆，无锁竞争。

---

## 7. LSH — 局部敏感哈希

> 源文件：[lsh.h](cpp/algorithms/lsh/lsh.h), [lsh.cpp](cpp/algorithms/lsh/lsh.cpp)

### 算法原理

局部敏感哈希（Locality-Sensitive Hashing）利用哈希函数的局部敏感性——相似向量以高概率映射到相同哈希桶。使用多个哈希表（num_hash_tables）提高召回率，每个表使用多个哈希函数（num_hash_functions）组合产生哈希值。搜索时只需检查查询哈希桶中的候选向量，将搜索范围从全库缩小到少量候选。

### 调优方法

#### 7.1 64字节对齐的哈希函数存储

**方法**：哈希函数矩阵使用`AlignedAllocator<float, 64>`分配。

```cpp
std::vector<float, AlignedAllocator<float, 64>> hash_functions_flat_;
```

**原理**：AVX-512的对齐加载指令`_mm512_load_ps`要求64字节对齐。对齐加载比非对齐加载`_mm512_loadu_ps`更快（某些微架构上差1~2个时钟周期），且对齐访问保证不跨越缓存行边界，避免性能惩罚。

#### 7.2 维度填充

**方法**：将向量维度向上取整到16的倍数。

```cpp
padded_dim_ = ((d + 15) / 16) * 16;
```

**原理**：AVX-512一次处理16个float，填充后无需处理尾部元素，消除循环中的条件判断和标量回退路径。填充部分置零不影响点积结果（零乘任何数为零）。

#### 7.3 SIMD加速的哈希计算

**方法**：使用AVX-512/AVX2加速随机投影点积计算。

```cpp
// AVX-512
for (size_t h = 0; h < num_hash_functions_; ++h) {
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < padded_dim_; i += 16) {
        __m512 v = _mm512_loadu_ps(vec + i);
        __m512 w = _mm512_load_ps(weights + i);  // 对齐加载
        sum = _mm512_fmadd_ps(v, w, sum);
    }
    float dot = _mm512_reduce_add_ps(sum) + bias;
    int bit = static_cast<int>(std::floor(dot * inv_r_));
    hash = (hash << 1) | (bit & 1);
}
```

**原理**：
- **FMA指令**：乘加融合减少指令数，提高吞吐。
- **`_mm512_reduce_add_ps`**：AVX-512内置的水平求和指令，比AVX2的手动shuffle+hadd更简洁高效。
- **对齐加载权重**：哈希函数矩阵预分配对齐内存，使用`_mm512_load_ps`对齐加载。

#### 7.4 多探针搜索

**方法**：`generate_probe_sequence`生成与原始哈希值汉明距离递增的探测序列。

```cpp
void generate_probe_sequence(size_t hash, size_t num_bits,
                              size_t max_probes, size_t* probes, size_t& n_probes) {
    // 按汉明距离从小到大翻转哈希位
    for (size_t mask = 1; mask < total_possible; ++mask) {
        size_t flipped = hash;
        int hd = 0;
        for (size_t bit = 0; bit < max_flips; ++bit) {
            if (mask & (1u << bit)) {
                flipped ^= (1u << (num_bits - 1 - bit));
                ++hd;
            }
        }
        entries.push_back({flipped, hd});
    }
    std::sort(entries.begin(), entries.end(), /* by hamming_dist */);
}
```

**原理**：
- **提高召回率**：单探针只检查精确匹配的桶，可能遗漏近邻。多探针检查相邻桶（汉明距离1、2、...的桶），显著提高召回率。
- **汉明距离排序**：汉明距离越小的桶，包含近邻的概率越高。按汉明距离从小到大探测，在有限探测次数内获得最大召回。
- **效率权衡**：增加探测次数提高召回但增加候选集大小和精确计算量。`num_probes`参数允许用户在速度和精度间权衡。

#### 7.5 递增访问标记去重

**方法**：与HNSW相同的递增标记技术。

```cpp
std::vector<int32_t> seen(ntotal, 0);
int32_t query_id = 0;
// 每次查询
++query_id;
if (seen[idx] != query_id) { seen[idx] = query_id; candidates.push_back(idx); }
```

**原理**：多表多探针搜索中，同一向量可能出现在多个桶中。递增标记实现O(1)去重，避免重复距离计算。

#### 7.6 最大堆Top-k + 距离截断

**方法**：与其他算法相同的最大堆top-k维护策略。

**原理**：候选集通常远大于k，截断策略使大部分候选仅需一次比较即可跳过。

---

## 优化技术总结

| 优化技术 | FlatL2 | FlatIP | HNSW | IVF | PQ | LSH |
|---------|--------|--------|------|-----|----|-----|
| SIMD距离计算(AVX2/AVX512) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 转置数据布局 | ✅ | ✅ | - | ✅ | ✅ | - |
| 软件预取 | ✅ | ✅ | - | - | - | - |
| 循环展开 | ✅ | ✅ | - | - | - | - |
| 多线程/OpenMP并行 | ✅ | ✅ | - | ✅ | ✅ | - |
| 自适应线程数 | ✅ | ✅ | - | - | - | - |
| 紧凑数据结构 | - | - | ✅ | - | ✅ | - |
| 预计算范数 | - | - | ✅ | ✅ | ✅ | - |
| 距离分解(范数+内积) | - | - | ✅ | ✅ | ✅ | - |
| 自定义堆操作 | - | - | ✅ | - | - | - |
| 提前终止/剪枝 | - | - | ✅ | ✅ | - | - |
| 递增访问标记 | - | - | ✅ | - | - | ✅ |
| K-Means++初始化 | - | - | - | ✅ | - | - |
| nth_element部分排序 | - | - | - | ✅ | - | - |
| Gather指令查表 | - | - | - | - | ✅ | - |
| 多探针搜索 | - | - | - | - | - | ✅ |
| 内存对齐分配 | - | - | - | - | - | ✅ |
| 维度填充 | - | - | - | - | - | ✅ |
| 栈分配小缓冲区 | ✅ | - | ✅ | - | ✅ | - |

---

## 优化层次分类

### 算法层优化
改变算法行为或数据结构以减少计算量：
- HNSW紧凑邻居存储、双堆剪枝、提前终止
- IVF的K-Means++、空聚类处理、nth_element
- PQ的距离表查找法、转置编码
- LSH的多探针搜索

### 系统层优化
利用硬件特性加速计算：
- SIMD向量化（AVX2/AVX512）
- 多线程并行（std::thread / OpenMP）
- 内存对齐（AlignedAllocator）
- 软件预取（__builtin_prefetch）

### 数据布局优化
改善内存访问模式以提高缓存命中率：
- 转置数据布局（FlatL2/FlatIP/PQ）
- 连续聚类布局（IVF）
- 紧凑邻居存储（HNSW）
- 维度填充（LSH）

### 计算层优化
减少冗余计算：
- 预计算范数（HNSW/IVF/PQ）
- 距离分解（HNSW/IVF/PQ）
- 递增访问标记（HNSW/LSH）
- 循环展开（FlatL2/FlatIP）
