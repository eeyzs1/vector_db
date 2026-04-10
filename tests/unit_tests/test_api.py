import pytest
from fastapi.testclient import TestClient
from vector_db.api.main import app

client = TestClient(app)

class TestAPI:
    def test_root_endpoint(self):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Vector Database API"}
    
    def test_process_text_endpoint(self):
        """测试文本处理端点"""
        response = client.post(
            "/api/process/text",
            json={"text": "这是一个测试文本"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "chunks" in data
    
    def test_insert_vector_endpoint(self):
        """测试向量插入端点"""
        response = client.post(
            "/api/vector/insert",
            json={
                "collection_id": "test_collection",
                "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "metadata": [{"id": 1}, {"id": 2}]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "vector_ids" in data
        assert len(data["vector_ids"]) == 2
    
    def test_search_vector_endpoint(self):
        """测试向量搜索端点"""
        response = client.post(
            "/api/vector/search",
            json={
                "collection_id": "test_collection",
                "query_vector": [0.1, 0.2, 0.3],
                "top_k": 5,
                "filters": {}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
    
    def test_file_upload_endpoint(self):
        """测试文件上传端点"""
        # 创建一个临时文本文件
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write("这是一个测试文件")
        
        try:
            with open("test.txt", "rb") as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": f},
                    data={"collection_id": "test_collection"}
                )
            assert response.status_code == 200
            data = response.json()
            assert "file_id" in data
            assert "collection_id" in data
            assert "chunks_processed" in data
            assert "vectors_stored" in data
        finally:
            # 清理临时文件
            import os
            if os.path.exists("test.txt"):
                os.remove("test.txt")
    
    def test_metadata_endpoints(self):
        """测试元数据管理端点"""
        # 首先上传一个文件获取file_id
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write("这是一个测试文件")
        
        try:
            with open("test.txt", "rb") as f:
                upload_response = client.post(
                    "/api/files/upload",
                    files={"file": f},
                    data={"collection_id": "test_collection"}
                )
            file_id = upload_response.json()["file_id"]
            
            # 测试获取元数据
            get_response = client.get(f"/api/metadata/{file_id}")
            assert get_response.status_code == 200
            
            # 测试更新元数据
            update_response = client.post(
                "/api/metadata/update",
                json={"file_id": file_id, "metadata": {"description": "测试文件"}}
            )
            assert update_response.status_code == 200
            
            # 测试删除元数据
            delete_response = client.delete(f"/api/metadata/{file_id}")
            assert delete_response.status_code == 200
        finally:
            # 清理临时文件
            import os
            if os.path.exists("test.txt"):
                os.remove("test.txt")