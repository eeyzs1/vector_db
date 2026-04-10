
from enum import Enum
from typing import Any, Tuple, Optional, Union, Dict, List
from concurrent.futures import ThreadPoolExecutor
import asyncio

import numpy as np
from numpy.typing import NDArray


class IndexType(Enum):
    FLAT_L2 = "flat_l2"
    FLAT_IP = "flat_ip"
    HNSW = "hnsw"
    IVF = "ivf"
    PQ = "pq"
    LSH = "lsh"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"
    ANNOY = "annoy"


class Implementation(Enum):
    CPP = "cpp"
    RUST = "rust"
    CPP_NANOBIND = "cpp_nanobind"


class VectorIndex:
    _shared_executor: Optional[ThreadPoolExecutor] = None

    def __init__(
        self,
        index_type: IndexType,
        dimension: int,
        implementation: Implementation = Implementation.CPP_NANOBIND,
        **kwargs: Any
    ):
        self.index_type: IndexType = index_type
        self.dimension: int = dimension
        self.implementation: Implementation = implementation
        self._index: Optional[Any] = None
        self._initialize_index(**kwargs)

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        if cls._shared_executor is None or cls._shared_executor._shutdown:
            cls._shared_executor = ThreadPoolExecutor(max_workers=4)
        return cls._shared_executor

    def _initialize_index(self, **kwargs: Any) -> None:
        if self.implementation == Implementation.CPP:
            self._init_cpp_index(**kwargs)
        elif self.implementation == Implementation.CPP_NANOBIND:
            self._init_cpp_nanobind_index(**kwargs)
        elif self.implementation == Implementation.RUST:
            self._init_rust_index(**kwargs)
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _init_cpp_index(self, **kwargs: Any) -> None:
        try:
            if self.index_type == IndexType.FLAT_L2:
                from .cpp import _flat
                self._index = _flat.IndexFlatL2(self.dimension)
            elif self.index_type == IndexType.FLAT_IP:
                from .cpp import _flat_ip
                self._index = _flat_ip.IndexFlatIP(self.dimension)
            elif self.index_type == IndexType.IVF:
                from .cpp import vectordb_ivf
                nlist = kwargs.get('nlist', 100)
                self._index = vectordb_ivf.IndexIVF(self.dimension, nlist)
            elif self.index_type == IndexType.HNSW:
                from .cpp import vectordb_hnsw
                M = kwargs.get('M', 16)
                ef_construction = kwargs.get('ef_construction', 200)
                self._index = vectordb_hnsw.IndexHNSW(self.dimension, M, ef_construction)
            elif self.index_type == IndexType.PQ:
                from .cpp import vectordb_pq
                M = kwargs.get('M', 8)
                nbits = kwargs.get('nbits', 8)
                self._index = vectordb_pq.IndexPQ(self.dimension, M, nbits)
            elif self.index_type == IndexType.LSH:
                from .cpp import vectordb_lsh
                num_hash_tables = kwargs.get('num_hash_tables', 8)
                num_hash_functions = kwargs.get('num_hash_functions', 4)
                r = kwargs.get('r', 1.0)
                self._index = vectordb_lsh.IndexLSH(self.dimension, num_hash_tables, num_hash_functions, r)
            elif self.index_type == IndexType.KD_TREE:
                from .cpp import vectordb_kdtree
                self._index = vectordb_kdtree.IndexKDTree(self.dimension)
            elif self.index_type == IndexType.BALL_TREE:
                from .cpp import vectordb_balltree
                leaf_size = kwargs.get('leaf_size', 40)
                self._index = vectordb_balltree.IndexBallTree(self.dimension, leaf_size)
            elif self.index_type == IndexType.ANNOY:
                from .cpp import vectordb_annoy
                n_trees = kwargs.get('n_trees', 10)
                self._index = vectordb_annoy.IndexAnnoy(self.dimension, n_trees)
            else:
                raise NotImplementedError(
                    f"Index type {self.index_type} not yet implemented in C++"
                )
        except ImportError as e:
            raise ImportError(
                f"Failed to import C++ implementation: {e}. "
                "Make sure the C++ extensions are compiled."
            )

    def _init_cpp_nanobind_index(self, **kwargs: Any) -> None:
        try:
            if self.index_type == IndexType.FLAT_L2:
                from .cpp import _flat_nanobind
                self._index = _flat_nanobind.IndexFlatL2(self.dimension)
            elif self.index_type == IndexType.FLAT_IP:
                from .cpp import _flat_ip_nanobind
                self._index = _flat_ip_nanobind.IndexFlatIP(self.dimension)
            elif self.index_type == IndexType.IVF:
                from .cpp import _ivf_nanobind
                nlist = kwargs.get('nlist', 100)
                self._index = _ivf_nanobind.IndexIVF(self.dimension, nlist)
            elif self.index_type == IndexType.HNSW:
                from .cpp import _hnsw_nanobind
                M = kwargs.get('M', 16)
                ef_construction = kwargs.get('ef_construction', 200)
                self._index = _hnsw_nanobind.IndexHNSW(self.dimension, M, ef_construction)
            elif self.index_type == IndexType.PQ:
                from .cpp import _pq_nanobind
                M = kwargs.get('M', 8)
                nbits = kwargs.get('nbits', 8)
                self._index = _pq_nanobind.IndexPQ(self.dimension, M, nbits)
            elif self.index_type == IndexType.LSH:
                from .cpp import _lsh_nanobind
                num_hash_tables = kwargs.get('num_hash_tables', 8)
                num_hash_functions = kwargs.get('num_hash_functions', 4)
                r = kwargs.get('r', 1.0)
                self._index = _lsh_nanobind.IndexLSH(self.dimension, num_hash_tables, num_hash_functions, r)
            elif self.index_type == IndexType.KD_TREE:
                from .cpp import _kdtree_nanobind
                leaf_size = kwargs.get('leaf_size', 40)
                self._index = _kdtree_nanobind.IndexKDTree(self.dimension, leaf_size)
            elif self.index_type == IndexType.BALL_TREE:
                from .cpp import _balltree_nanobind
                leaf_size = kwargs.get('leaf_size', 40)
                self._index = _balltree_nanobind.IndexBallTree(self.dimension, leaf_size)
            elif self.index_type == IndexType.ANNOY:
                from .cpp import _annoy_nanobind
                n_trees = kwargs.get('n_trees', 10)
                self._index = _annoy_nanobind.IndexAnnoy(self.dimension, n_trees)
            else:
                raise NotImplementedError(
                    f"Index type {self.index_type} not yet implemented in C++ nanobind"
                )
        except ImportError as e:
            raise ImportError(
                f"Failed to import C++ nanobind implementation: {e}. "
                "Make sure the C++ nanobind extensions are compiled."
            )

    def _init_rust_index(self, **kwargs: Any) -> None:
        try:
            if self.index_type == IndexType.FLAT_L2:
                from .rust import _flat
                self._index = _flat.FlatIndex(self.dimension)
            elif self.index_type == IndexType.FLAT_IP:
                from .rust import _flat_ip
                self._index = _flat_ip.FlatIPIndex(self.dimension)
            elif self.index_type == IndexType.HNSW:
                from .rust import vectordb_hnsw
                M = kwargs.get('M', 16)
                ef_construction = kwargs.get('ef_construction', 200)
                self._index = vectordb_hnsw.IndexHNSW(self.dimension, M, ef_construction)
            elif self.index_type == IndexType.IVF:
                from .rust import vectordb_ivf
                nlist = kwargs.get('nlist', 100)
                self._index = vectordb_ivf.IndexIVF(self.dimension, nlist)
            elif self.index_type == IndexType.PQ:
                from .rust import vectordb_pq
                M = kwargs.get('M', 8)
                nbits = kwargs.get('nbits', 8)
                self._index = vectordb_pq.IndexPQ(self.dimension, M, nbits)
            elif self.index_type == IndexType.LSH:
                from .rust import vectordb_lsh
                num_hash_tables = kwargs.get('num_hash_tables', 8)
                num_hash_functions = kwargs.get('num_hash_functions', 4)
                r = kwargs.get('r', 1.0)
                self._index = vectordb_lsh.IndexLSH(self.dimension, num_hash_tables, num_hash_functions, r)
            elif self.index_type == IndexType.KD_TREE:
                from .rust import vectordb_kdtree
                leaf_size = kwargs.get('leaf_size', 40)
                self._index = vectordb_kdtree.IndexKDTree(self.dimension, leaf_size)
            elif self.index_type == IndexType.BALL_TREE:
                from .rust import vectordb_balltree
                leaf_size = kwargs.get('leaf_size', 40)
                self._index = vectordb_balltree.IndexBallTree(self.dimension, leaf_size)
            elif self.index_type == IndexType.ANNOY:
                from .rust import vectordb_annoy
                n_trees = kwargs.get('n_trees', 10)
                self._index = vectordb_annoy.IndexAnnoy(self.dimension, n_trees)
            else:
                raise NotImplementedError(
                    f"Index type {self.index_type} not yet implemented in Rust"
                )
        except ImportError as e:
            raise ImportError(
                f"Failed to import Rust implementation: {e}. "
                "Make sure the Rust extensions are compiled."
            )

    def train(self, vectors: NDArray[np.float32]) -> None:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if hasattr(self._index, 'train'):
            self._index.train(vectors)

    async def train_async(self, vectors: NDArray[np.float32]) -> None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            self.train,
            vectors
        )

    def add(self, vectors: NDArray[np.float32]) -> None:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if hasattr(self._index, 'add_buf'):
            self._index.add_buf(vectors)
        else:
            self._index.add(vectors)

    async def add_async(self, vectors: NDArray[np.float32]) -> None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            self.add,
            vectors
        )

    def build(self) -> None:
        if hasattr(self._index, 'build'):
            self._index.build()

    def set_nprobe(self, nprobe: int) -> None:
        if hasattr(self._index, 'set_nprobe'):
            self._index.set_nprobe(nprobe)

    def set_ef_search(self, ef: int) -> None:
        if hasattr(self._index, 'set_ef_search'):
            self._index.set_ef_search(ef)

    def set_search_k(self, search_k: int) -> None:
        if hasattr(self._index, 'set_search_k'):
            self._index.set_search_k(search_k)

    def search(
        self,
        queries: NDArray[np.float32],
        k: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        if queries.ndim != 2:
            raise ValueError("Queries must be a 2D array")
        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {queries.shape[1]}"
            )

        if hasattr(self._index, 'search_batch_buf'):
            distances, labels = self._index.search_batch_buf(queries, k)
            return np.array(distances, dtype=np.float32), np.array(labels, dtype=np.int64)

        if hasattr(self._index, 'search'):
            if queries.shape[0] == 1:
                distances, labels = self._index.search(queries[0], k)
                return np.array([distances], dtype=np.float32), np.array([labels], dtype=np.int64)
            else:
                all_distances: List[List[float]] = []
                all_labels: List[List[int]] = []
                for query in queries:
                    dist, lbl = self._index.search(query, k)
                    all_distances.append(dist)
                    all_labels.append(lbl)
                return np.array(all_distances, dtype=np.float32), np.array(all_labels, dtype=np.int64)

        raise NotImplementedError("Search not implemented for this index")

    async def search_async(
        self,
        queries: NDArray[np.float32],
        k: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            self.search,
            queries,
            k
        )

    @property
    def ntotal(self) -> int:
        if hasattr(self._index, 'ntotal'):
            return self._index.ntotal
        elif hasattr(self._index, 'size'):
            return self._index.size()
        else:
            raise NotImplementedError("Size not available for this index")

    def size(self) -> int:
        return self.ntotal

    def get_dimension(self) -> int:
        return self.dimension

    def close(self) -> None:
        pass

    def __enter__(self) -> 'VectorIndex':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def create_index(
    index_type: str,
    dimension: int,
    implementation: str = "cpp_nanobind",
    **kwargs: Any
) -> VectorIndex:
    idx_type = IndexType(index_type.lower())
    impl = Implementation(implementation.lower())
    return VectorIndex(idx_type, dimension, impl, **kwargs)
