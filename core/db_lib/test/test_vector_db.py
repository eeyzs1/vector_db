
"""
Comprehensive test suite for 7VecDB
Includes unit tests, correctness tests (against FAISS), and boundary condition tests
"""

import pytest
import numpy as np
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from vectordb import VectorIndex, IndexType, Implementation, create_index

# Try to import FAISS for correctness testing
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available, some correctness tests will be skipped")


class TestVectorIndexCreation:
    """Test creation of various index types"""
    
    @pytest.mark.parametrize("index_type", [IndexType.FLAT_L2, IndexType.FLAT_IP])
    @pytest.mark.parametrize("implementation", [Implementation.CPP, Implementation.RUST])
    def test_basic_index_creation(self, index_type, implementation):
        """Test that we can create basic index types"""
        dimension = 128
        index = VectorIndex(index_type, dimension, implementation)
        assert index.get_dimension() == dimension
        assert index.size() == 0
        index.close()
    
    def test_create_index_function(self):
        """Test the create_index convenience function"""
        index = create_index("flat_l2", 64, "cpp")
        assert index.get_dimension() == 64
        assert index.size() == 0
        index.close()
    
    @pytest.mark.parametrize("implementation", [Implementation.CPP, Implementation.RUST])
    def test_context_manager(self, implementation):
        """Test that context manager works properly"""
        with create_index("flat_l2", 32, implementation.value) as index:
            assert index.size() == 0
            vectors = np.random.rand(10, 32).astype(np.float32)
            index.add(vectors)
            assert index.size() == 10


class TestBasicOperations:
    """Test basic vector index operations"""
    
    @pytest.fixture
    def sample_vectors(self):
        """Fixture providing sample vectors"""
        np.random.seed(42)
        return np.random.rand(100, 128).astype(np.float32)
    
    @pytest.fixture
    def sample_queries(self):
        """Fixture providing sample queries"""
        np.random.seed(43)
        return np.random.rand(5, 128).astype(np.float32)
    
    @pytest.mark.parametrize("index_type", ["flat_l2", "flat_ip"])
    @pytest.mark.parametrize("implementation", ["cpp", "rust"])
    def test_add_vectors(self, index_type, implementation, sample_vectors):
        """Test adding vectors to index"""
        index = create_index(index_type, 128, implementation)
        assert index.size() == 0
        
        index.add(sample_vectors)
        assert index.size() == sample_vectors.shape[0]
        
        index.close()
    
    @pytest.mark.parametrize("index_type", ["flat_l2", "flat_ip"])
    @pytest.mark.parametrize("implementation", ["cpp", "rust"])
    def test_search_vectors(self, index_type, implementation, sample_vectors, sample_queries):
        """Test searching vectors"""
        index = create_index(index_type, 128, implementation)
        index.add(sample_vectors)
        
        k = 5
        distances, labels = index.search(sample_queries, k)
        
        assert distances.shape == (sample_queries.shape[0], k)
        assert labels.shape == (sample_queries.shape[0], k)
        
        # Check that labels are valid indices
        assert np.all(labels >= 0)
        assert np.all(labels &lt; sample_vectors.shape[0])
        
        index.close()


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
class TestCorrectnessAgainstFAISS:
    """Test correctness by comparing results with FAISS"""
    
    @pytest.fixture
    def test_data(self):
        """Fixture providing test data"""
        np.random.seed(42)
        num_vectors = 1000
        num_queries = 10
        dimension = 64
        
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        queries = np.random.rand(num_queries, dimension).astype(np.float32)
        
        return vectors, queries
    
    def test_flat_l2_correctness_cpp(self, test_data):
        """Test Flat L2 correctness (C++ implementation vs FAISS)"""
        vectors, queries = test_data
        dimension = vectors.shape[1]
        k = 10
        
        # Create and populate our index
        our_index = create_index("flat_l2", dimension, "cpp")
        our_index.add(vectors)
        our_distances, our_labels = our_index.search(queries, k)
        our_index.close()
        
        # Create and populate FAISS index
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(vectors)
        faiss_distances, faiss_labels = faiss_index.search(queries, k)
        
        # Compare results - for exact search, labels should match
        np.testing.assert_array_equal(our_labels, faiss_labels)
        # Distances should be very close
        np.testing.assert_allclose(our_distances, faiss_distances, rtol=1e-5)
    
    def test_flat_l2_correctness_rust(self, test_data):
        """Test Flat L2 correctness (Rust implementation vs FAISS)"""
        vectors, queries = test_data
        dimension = vectors.shape[1]
        k = 10
        
        # Create and populate our index
        our_index = create_index("flat_l2", dimension, "rust")
        our_index.add(vectors)
        our_distances, our_labels = our_index.search(queries, k)
        our_index.close()
        
        # Create and populate FAISS index
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(vectors)
        faiss_distances, faiss_labels = faiss_index.search(queries, k)
        
        # Compare results - for exact search, labels should match
        np.testing.assert_array_equal(our_labels, faiss_labels)
        # Distances should be very close
        np.testing.assert_allclose(our_distances, faiss_distances, rtol=1e-5)
    
    def test_flat_ip_correctness_cpp(self, test_data):
        """Test Flat IP correctness (C++ implementation vs FAISS)"""
        vectors, queries = test_data
        dimension = vectors.shape[1]
        k = 10
        
        # Create and populate our index
        our_index = create_index("flat_ip", dimension, "cpp")
        our_index.add(vectors)
        our_distances, our_labels = our_index.search(queries, k)
        our_index.close()
        
        # Create and populate FAISS index
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(vectors)
        faiss_distances, faiss_labels = faiss_index.search(queries, k)
        
        # Compare results - for exact search, labels should match
        np.testing.assert_array_equal(our_labels, faiss_labels)
        # Distances should be very close (note: FAISS returns similarity, not distance)
        # For IP, higher = better, so we need to check the ordering
        np.testing.assert_allclose(our_distances, -faiss_distances if our_distances[0, 0] &lt; 0 else our_distances, rtol=1e-5)


class TestBoundaryConditions:
    """Test boundary conditions and edge cases"""
    
    def test_zero_vectors(self):
        """Test searching with zero vectors in index"""
        index = create_index("flat_l2", 32, "cpp")
        queries = np.random.rand(1, 32).astype(np.float32)
        
        with pytest.raises(Exception):
            index.search(queries, k=5)
        
        index.close()
    
    def test_k_greater_than_num_vectors(self):
        """Test k greater than number of vectors"""
        index = create_index("flat_l2", 32, "cpp")
        vectors = np.random.rand(5, 32).astype(np.float32)
        index.add(vectors)
        queries = np.random.rand(1, 32).astype(np.float32)
        
        # Should handle this gracefully (return all 5 vectors)
        distances, labels = index.search(queries, k=10)
        assert distances.shape == (1, 5)
        assert labels.shape == (1, 5)
        
        index.close()
    
    def test_dimension_mismatch_add(self):
        """Test adding vectors with wrong dimension"""
        index = create_index("flat_l2", 32, "cpp")
        vectors = np.random.rand(10, 64).astype(np.float32)
        
        with pytest.raises(ValueError):
            index.add(vectors)
        
        index.close()
    
    def test_dimension_mismatch_search(self):
        """Test searching with queries of wrong dimension"""
        index = create_index("flat_l2", 32, "cpp")
        vectors = np.random.rand(10, 32).astype(np.float32)
        index.add(vectors)
        queries = np.random.rand(1, 64).astype(np.float32)
        
        with pytest.raises(ValueError):
            index.search(queries, k=5)
        
        index.close()
    
    def test_single_vector(self):
        """Test with single vector"""
        index = create_index("flat_l2", 32, "cpp")
        vector = np.random.rand(1, 32).astype(np.float32)
        index.add(vector)
        
        # Search for the same vector
        distances, labels = index.search(vector, k=1)
        
        assert labels[0, 0] == 0
        # Distance should be very close to zero
        assert distances[0, 0] &lt; 1e-6
        
        index.close()
    
    def test_multiple_adds(self):
        """Test adding vectors in multiple batches"""
        index = create_index("flat_l2", 32, "cpp")
        
        vectors1 = np.random.rand(50, 32).astype(np.float32)
        vectors2 = np.random.rand(50, 32).astype(np.float32)
        
        index.add(vectors1)
        assert index.size() == 50
        
        index.add(vectors2)
        assert index.size() == 100
        
        index.close()
    
    def test_empty_queries(self):
        """Test with empty query array"""
        index = create_index("flat_l2", 32, "cpp")
        vectors = np.random.rand(10, 32).astype(np.float32)
        index.add(vectors)
        
        queries = np.empty((0, 32), dtype=np.float32)
        distances, labels = index.search(queries, k=5)
        
        assert distances.shape == (0, 5)
        assert labels.shape == (0, 5)
        
        index.close()


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations"""
    
    @pytest.fixture
    def sample_vectors(self):
        """Fixture providing sample vectors"""
        np.random.seed(42)
        return np.random.rand(100, 64).astype(np.float32)
    
    @pytest.fixture
    def sample_queries(self):
        """Fixture providing sample queries"""
        np.random.seed(43)
        return np.random.rand(5, 64).astype(np.float32)
    
    async def test_async_search(self, sample_vectors, sample_queries):
        """Test async search"""
        index = create_index("flat_l2", 64, "cpp")
        index.add(sample_vectors)
        
        k = 5
        distances, labels = await index.search_async(sample_queries, k)
        
        assert distances.shape == (sample_queries.shape[0], k)
        assert labels.shape == (sample_queries.shape[0], k)
        
        index.close()

