
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "../algorithms/flat/flat.h"

using namespace vectordb::algorithms;

class FlatL2Test : public ::testing::Test {
protected:
    void SetUp() override {
        dimension = 128;
        index = new IndexFlatL2(dimension);
    }

    void TearDown() override {
        delete index;
    }

    std::vector<float> generate_random_vectors(size_t num_vectors, size_t dim) {
        std::vector<float> vectors(num_vectors * dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (size_t i = 0; i < num_vectors * dim; ++i) {
            vectors[i] = static_cast<float>(dis(gen));
        }
        vectordb::normalize_vectors(vectors.data(), num_vectors, dim);
        return vectors;
    }

    size_t dimension;
    IndexFlatL2* index;
};

TEST_F(FlatL2Test, CanCreateIndex) {
    EXPECT_NE(index, nullptr);
}

TEST_F(FlatL2Test, CanAddVectors) {
    size_t num_vectors = 100;
    auto vectors = generate_random_vectors(num_vectors, dimension);
    
    // Note: Since we don't have a public size() method,
    // we'll just verify that add doesn't crash
    index->add(num_vectors, vectors.data());
    
    // If we reach here, the test passes
    SUCCEED();
}

TEST_F(FlatL2Test, CanSearchVectors) {
    size_t num_vectors = 100;
    auto vectors = generate_random_vectors(num_vectors, dimension);
    index->add(num_vectors, vectors.data());
    
    size_t k = 5;
    std::vector<float> distances(k);
    std::vector<size_t> labels(k);
    
    // Search for the first vector
    index->search(1, vectors.data(), k, distances.data(), labels.data());
    
    // The first result should be the vector itself
    EXPECT_EQ(labels[0], 0);
    EXPECT_LT(distances[0], 1e-6);
}

TEST_F(FlatL2Test, SearchReturnsValidLabels) {
    size_t num_vectors = 100;
    auto vectors = generate_random_vectors(num_vectors, dimension);
    index->add(num_vectors, vectors.data());
    
    size_t k = 10;
    auto query = generate_random_vectors(1, dimension);
    std::vector<float> distances(k);
    std::vector<size_t> labels(k);
    
    index->search(1, query.data(), k, distances.data(), labels.data());
    
    for (size_t i = 0; i < k; ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], num_vectors);
    }
}

TEST_F(FlatL2Test, DistancesAreSorted) {
    size_t num_vectors = 100;
    auto vectors = generate_random_vectors(num_vectors, dimension);
    index->add(num_vectors, vectors.data());
    
    size_t k = 10;
    auto query = generate_random_vectors(1, dimension);
    std::vector<float> distances(k);
    std::vector<size_t> labels(k);
    
    index->search(1, query.data(), k, distances.data(), labels.data());
    
    for (size_t i = 1; i < k; ++i) {
        EXPECT_GE(distances[i], distances[i-1]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

