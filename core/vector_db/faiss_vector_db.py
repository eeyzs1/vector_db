import faiss
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from filelock import FileLock
from .base_vector_db import BaseVectorDB

class FAISSVectorDB(BaseVectorDB):
    def __init__(self, db_path='./data/vector_db'):
        self.db_path = db_path
        self.lock_dir = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.collections = {}   # collection_id -> faiss index
        self.metadata = {}      # collection_id -> {vector_id: metadata}
        self.index_to_id = {}   # collection_id -> [vector_id, ...] ordered by faiss position
        self._load_collections()

    def _get_lock(self, collection_id: str) -> FileLock:
        return FileLock(os.path.join(self.lock_dir, f"{collection_id}.lock"))

    def _load_collections(self):
        for file_name in os.listdir(self.db_path):
            if file_name.endswith('.faiss'):
                collection_id = file_name[:-6]
                index_path = os.path.join(self.db_path, file_name)
                metadata_path = os.path.join(self.db_path, f"{collection_id}.pkl")

                index = faiss.read_index(index_path)
                self.collections[collection_id] = index

                with open(metadata_path, 'rb') as f:
                    saved = pickle.load(f)
                # support both old format (dict only) and new format (dict + index_to_id)
                if isinstance(saved, tuple):
                    self.metadata[collection_id], self.index_to_id[collection_id] = saved
                else:
                    self.metadata[collection_id] = saved
                    self.index_to_id[collection_id] = list(saved.keys())
    
    def create_collection(self, collection_id: str, dimension: int) -> bool:
        if collection_id in self.collections:
            return False

        with self._get_lock(collection_id):
            index = faiss.IndexFlatL2(dimension)
            self.collections[collection_id] = index
            self.metadata[collection_id] = {}
            self.index_to_id[collection_id] = []

            try:
                self._save_collection(collection_id)
                return True
            except Exception as e:
                print(f"Create collection error: {str(e)}")
                return False
    
    def insert(self, collection_id: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")

        import uuid
        with self._get_lock(collection_id):
            index = self.collections[collection_id]
            vector_ids = []
            try:
                for i, vector in enumerate(vectors):
                    vector_id = str(uuid.uuid4())
                    vector_ids.append(vector_id)
                    index.add(np.array([vector], dtype=np.float32))
                    meta_with_vector = metadata[i].copy()
                    meta_with_vector['vector'] = vector
                    self.metadata[collection_id][vector_id] = meta_with_vector
                    self.index_to_id[collection_id].append(vector_id)

                self._save_collection(collection_id)
                return vector_ids
            except Exception as e:
                print(f"Insert error: {str(e)}")
                raise
    
    def batch_insert(self, collection_id: str, vectors: List[List[float]], metadata: List[Dict[str, Any]], batch_size: int = 1000) -> List[str]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")

        import uuid
        with self._get_lock(collection_id):
            index = self.collections[collection_id]
            vector_ids = []
            try:
                for i in range(0, len(vectors), batch_size):
                    batch_vectors = vectors[i:i+batch_size]
                    batch_metadata = metadata[i:i+batch_size]
                    batch_ids = [str(uuid.uuid4()) for _ in batch_vectors]

                    vectors_np = np.array(batch_vectors, dtype=np.float32)
                    index.add(vectors_np)

                    for vector_id, meta, vector in zip(batch_ids, batch_metadata, batch_vectors):
                        meta_with_vector = meta.copy()
                        meta_with_vector['vector'] = vector
                        self.metadata[collection_id][vector_id] = meta_with_vector
                        self.index_to_id[collection_id].append(vector_id)

                    vector_ids.extend(batch_ids)

                self._save_collection(collection_id)
                return vector_ids
            except Exception as e:
                print(f"Batch insert error: {str(e)}")
                raise
    
    def search(self, collection_id: str, query_vector: List[float], top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")

        index = self.collections[collection_id]

        try:
            distances, indices = index.search(np.array([query_vector], dtype=np.float32), top_k)

            results = []
            id_list = self.index_to_id[collection_id]
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(id_list):
                    vector_id = id_list[idx]
                    if vector_id in self.metadata[collection_id]:
                        metadata = self.metadata[collection_id][vector_id]
                        results.append({
                            'vector_id': vector_id,
                            'distance': float(distances[0][i]),
                            'metadata': metadata
                        })

            if filters:
                results = [r for r in results if all(r['metadata'].get(k) == v for k, v in filters.items())]

            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            raise
    
    def modify(self, collection_id: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        if vector_id not in self.metadata[collection_id]:
            return False

        with self._get_lock(collection_id):
            meta_with_vector = metadata.copy()
            meta_with_vector['vector'] = vector
            self.metadata[collection_id][vector_id] = meta_with_vector
            self._rebuild_index_from_metadata(collection_id)
            self._save_collection(collection_id)
        return True

    def delete(self, collection_id: str, vector_ids: List[str]) -> bool:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")

        with self._get_lock(collection_id):
            deleted = False
            for vector_id in vector_ids:
                if vector_id in self.metadata[collection_id]:
                    del self.metadata[collection_id][vector_id]
                    if vector_id in self.index_to_id[collection_id]:
                        self.index_to_id[collection_id].remove(vector_id)
                    deleted = True

            if deleted:
                self._rebuild_index_from_metadata(collection_id)
                self._save_collection(collection_id)
        return deleted

    def _rebuild_index_from_metadata(self, collection_id: str):
        """从元数据重建 FAISS 索引，保持 index_to_id 顺序一致"""
        old_index = self.collections[collection_id]
        dimension = old_index.d
        new_index = faiss.IndexFlatL2(dimension)

        new_id_list = []
        vectors = []
        for vector_id in self.index_to_id[collection_id]:
            if vector_id in self.metadata[collection_id]:
                meta = self.metadata[collection_id][vector_id]
                if 'vector' in meta:
                    vectors.append(meta['vector'])
                    new_id_list.append(vector_id)

        if vectors:
            new_index.add(np.array(vectors, dtype=np.float32))

        self.collections[collection_id] = new_index
        self.index_to_id[collection_id] = new_id_list
    
    def _save_collection(self, collection_id: str):
        index_path = os.path.join(self.db_path, f"{collection_id}.faiss")
        faiss.write_index(self.collections[collection_id], index_path)

        metadata_path = os.path.join(self.db_path, f"{collection_id}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump((self.metadata[collection_id], self.index_to_id[collection_id]), f)
    
    def optimize_index(self, collection_id: str, nlist: int = 100) -> bool:
        """优化索引，使用IVF索引提高搜索速度"""
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")

        with self._get_lock(collection_id):
            try:
                index = self.collections[collection_id]
                dimension = index.d
                nlist = min(nlist, index.ntotal)
                if nlist < 1:
                    nlist = 1
                quantizer = faiss.IndexFlatL2(dimension)
                ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                vectors = []
                for i in range(index.ntotal):
                    vector = index.reconstruct(i)
                    vectors.append(vector.tolist())
                if vectors:
                    vectors_np = np.array(vectors, dtype=np.float32)
                    ivf_index.train(vectors_np)
                    ivf_index.add(vectors_np)
                    self.collections[collection_id] = ivf_index
                    self._save_collection(collection_id)
                    return True
                return False
            except Exception as e:
                print(f"Optimize index error: {str(e)}")
                raise

    def rebuild_index(self, collection_id: str) -> bool:
        """重建索引"""
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")

        with self._get_lock(collection_id):
            try:
                index = self.collections[collection_id]
                dimension = index.d
                new_index = faiss.IndexFlatL2(dimension)
                vectors = []
                metadata_keys = list(self.metadata[collection_id].keys())
                if hasattr(index, 'ntotal') and index.ntotal > 0:
                    if metadata_keys and 'vector' in self.metadata[collection_id][metadata_keys[0]]:
                        for vector_id in metadata_keys:
                            if 'vector' in self.metadata[collection_id][vector_id]:
                                vectors.append(self.metadata[collection_id][vector_id]['vector'])
                    else:
                        try:
                            for i in range(index.ntotal):
                                vector = index.reconstruct(i)
                                vectors.append(vector)
                        except RuntimeError:
                            return False
                if vectors:
                    vectors_np = np.array(vectors, dtype=np.float32)
                    new_index.add(vectors_np)
                    self.collections[collection_id] = new_index
                    self._save_collection(collection_id)
                    return True
                return False
            except Exception as e:
                print(f"Rebuild index error: {str(e)}")
                return False
    
    def delete_collection(self, collection_id: str) -> bool:
        if collection_id not in self.collections:
            return False

        with self._get_lock(collection_id):
            index_path = os.path.join(self.db_path, f"{collection_id}.faiss")
            if os.path.exists(index_path):
                os.remove(index_path)

            metadata_path = os.path.join(self.db_path, f"{collection_id}.pkl")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            del self.collections[collection_id]
            del self.metadata[collection_id]
            del self.index_to_id[collection_id]
        return True