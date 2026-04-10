import sys
import os
import json
from tests.performance_test import PerformanceTester, generate_test_vectors, generate_test_metadata
from tests.performance_config import PerformanceConfig

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_db_client(db_type):
    """获取数据库客户端"""
    config = PerformanceConfig.VECTOR_DB_CONFIGS.get(db_type)
    if not config:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    if db_type == "faiss":
        from core.vector_db.faiss_vector_db import FAISSVectorDB
        return FAISSVectorDB(db_path=config['path'])
    elif db_type == "hnsw":
        from core.vector_db.hnsw_vector_db import HNSWVectorDB
        return HNSWVectorDB(db_path=config['path'])
    elif db_type == "annoy":
        from core.vector_db.annoy_vector_db import AnnoyVectorDB
        return AnnoyVectorDB(db_path=config['path'])
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def run_tests():
    """运行所有性能测试"""
    # 确保目录存在
    PerformanceConfig.ensure_directories()
    
    # 测试所有数据库后端
    db_types = list(PerformanceConfig.VECTOR_DB_CONFIGS.keys())
    
    for db_type in db_types:
        print(f"\n=== Testing {db_type.upper()} backend ===")
        
        # 获取数据库客户端
        db_client = get_db_client(db_type)
        
        # 初始化性能测试器
        tester = PerformanceTester(db_client, f"7VecDB-{db_type.upper()}")
        
        # 运行测试
        for dimension in PerformanceConfig.DIMENSIONS:
            for size in PerformanceConfig.TEST_SIZES:
                print(f"\n--- Testing {size} vectors with {dimension} dimensions ---")
                
                try:
                    # 生成测试数据
                    vectors = generate_test_vectors(size, dimension)
                    metadata = generate_test_metadata(size)
                    
                    # 测试插入性能
                    tester.test_insert(vectors, metadata, batch_size=PerformanceConfig.BATCH_SIZE)
                    
                    # 生成查询向量
                    query_vectors = generate_test_vectors(PerformanceConfig.NUM_QUERY_VECTORS, dimension)
                    
                    # 测试搜索性能
                    for top_k in PerformanceConfig.TOP_K_VALUES:
                        tester.test_search(query_vectors, top_k)
                    
                    # 测试并发搜索
                    for concurrency in PerformanceConfig.CONCURRENCY_LEVELS:
                        # 限制查询向量数量以避免内存问题
                        test_query_vectors = query_vectors[:min(100, PerformanceConfig.NUM_QUERY_VECTORS)]
                        tester.test_concurrent_search(test_query_vectors, 10, concurrency)
                    
                    # 测试索引构建性能
                    tester.test_index_build(vectors[:10000])  # 使用10K向量测试索引构建
                    
                    # 保存结果
                    tester.save_results(f"{size}_{dimension}")
                    
                except Exception as e:
                    print(f"Error during testing: {str(e)}")
                finally:
                    # 清理测试集合
                    try:
                        db_client.delete_collection("test_collection")
                    except:
                        pass

def generate_report():
    """生成性能报告"""
    results_dir = PerformanceConfig.RESULTS_DIR
    report = {}
    
    # 收集所有测试结果
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 解析文件名获取数据库类型和测试参数
            parts = filename.split('_')
            db_type = parts[0]
            test_size = int(parts[1])
            dimension = int(parts[2].split('.')[0])
            
            if db_type not in report:
                report[db_type] = {}
            if test_size not in report[db_type]:
                report[db_type][test_size] = {}
            report[db_type][test_size][dimension] = data
    
    # 生成报告文件
    report_path = os.path.join(results_dir, "performance_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance report generated at {report_path}")
    return report

def generate_summary_report():
    """生成摘要报告"""
    report = generate_report()
    summary = {}
    
    for db_type, sizes in report.items():
        summary[db_type] = {}
        for size, dimensions in sizes.items():
            summary[db_type][size] = {}
            for dimension, data in dimensions.items():
                summary[db_type][size][dimension] = {
                    "insert_throughput": data.get('insert', {}).get('throughput', 0),
                    "search_qps": data.get('search', {}).get('qps', 0),
                    "index_build_time": data.get('index_build', {}).get('time', 0),
                    "concurrent_qps": data.get('concurrent_search', {}).get('qps', 0)
                }
    
    # 保存摘要报告
    summary_path = os.path.join(PerformanceConfig.RESULTS_DIR, "performance_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Performance summary generated at {summary_path}")
    return summary

if __name__ == "__main__":
    # 运行测试
    run_tests()
    
    # 生成报告
    generate_summary_report()