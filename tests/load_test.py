from locust import HttpUser, task, between
import json
import time

class VectorDBUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # 初始化测试数据
        self.test_text = "This is a test text for vector database load testing. It contains multiple sentences to test the embedding process."
        self.collection_id = "test_collection"
    
    @task
    def test_text_processing(self):
        # 测试文本处理
        start_time = time.time()
        response = self.client.post("/api/process/text", json={
            "text": self.test_text
        })
        end_time = time.time()
        print(f"Text processing time: {end_time - start_time:.4f}s")
    
    @task
    def test_vector_insert(self):
        # 测试向量插入
        start_time = time.time()
        response = self.client.post("/api/vector/insert", json={
            "collection_id": self.collection_id,
            "vectors": [[0.1] * 384],  # 384维向量
            "metadata": [{"file_id": "test", "file_type": "text"}]
        })
        end_time = time.time()
        print(f"Vector insert time: {end_time - start_time:.4f}s")
    
    @task
    def test_vector_search(self):
        # 测试向量搜索
        start_time = time.time()
        response = self.client.post("/api/vector/search", json={
            "collection_id": self.collection_id,
            "query_vector": [0.1] * 384,
            "top_k": 5
        })
        end_time = time.time()
        print(f"Vector search time: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    import os
    import subprocess
    # 创建测试结果目录
    os.makedirs("tests/loadtest", exist_ok=True)
    # 启动locust测试，设置执行时间为60秒，无界面模式运行并输出结果到CSV文件
    subprocess.run(["locust", "-f", __file__, "--host", "http://localhost:8001", "--run-time", "60s", "--users", "10", "--spawn-rate", "2", "--headless", "--csv", "tests/loadtest_results/test_results"])