import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 性能结果目录
results_dir = "tests/performance_results"

# 数据集和向量数量
datasets = ["sift1m", "gist1m"]
max_vectors_list = [5000, 50000]

# 数据库列表
databases = ["7VecDB-FAISS", "ChromaDB"]

# 性能指标
metrics = [
    "insert_throughput",
    "search_qps_k1",
    "search_latency_k1",
    "recall_k1",
    "concurrent_qps_5"
]

# 指标名称映射
metric_names = {
    "insert_throughput": "插入吞吐量 (vectors/sec)",
    "search_qps_k1": "搜索QPS (k=1)",
    "search_latency_k1": "平均延迟 (ms)",
    "recall_k1": "召回率 (k=1)",
    "concurrent_qps_5": "并发QPS (5线程)"
}

# 加载所有测试结果
results = {}
for dataset in datasets:
    results[dataset] = {}
    for max_vectors in max_vectors_list:
        results[dataset][max_vectors] = {}
        for db in databases:
            file_name = f"{db}_{dataset}_{max_vectors}.json"
            file_path = os.path.join(results_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    results[dataset][max_vectors][db] = json.load(f)
            else:
                print(f"File not found: {file_path}")

# 生成对比图表
for dataset in datasets:
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # 为每个数据库创建数据点
        for db in databases:
            x = []
            y = []
            for max_vectors in max_vectors_list:
                if db in results[dataset][max_vectors]:
                    x.append(max_vectors)
                    # 根据指标类型获取正确的数据
                    if metric == "insert_throughput":
                        if "insert" in results[dataset][max_vectors][db] and "throughput" in results[dataset][max_vectors][db]["insert"]:
                            y.append(results[dataset][max_vectors][db]["insert"]["throughput"])
                        else:
                            y.append(0)
                    elif metric == "search_qps_k1":
                        if "search" in results[dataset][max_vectors][db] and "qps" in results[dataset][max_vectors][db]["search"]:
                            y.append(results[dataset][max_vectors][db]["search"]["qps"])
                        else:
                            y.append(0)
                    elif metric == "search_latency_k1":
                        if "search" in results[dataset][max_vectors][db] and "avg_latency" in results[dataset][max_vectors][db]["search"]:
                            y.append(results[dataset][max_vectors][db]["search"]["avg_latency"])
                        else:
                            y.append(0)
                    elif metric == "recall_k1":
                        if "search" in results[dataset][max_vectors][db] and "recall" in results[dataset][max_vectors][db]["search"]:
                            y.append(results[dataset][max_vectors][db]["search"]["recall"])
                        else:
                            y.append(0)
                    elif metric == "concurrent_qps_5":
                        if "concurrent_search" in results[dataset][max_vectors][db] and "qps" in results[dataset][max_vectors][db]["concurrent_search"]:
                            y.append(results[dataset][max_vectors][db]["concurrent_search"]["qps"])
                        else:
                            y.append(0)
                    else:
                        y.append(0)
            
            # 绘制折线图
            plt.plot(x, y, marker='o', label=db)
        
        # 设置图表标题和标签
        plt.title(f"{dataset} 数据集 - {metric_names[metric]}")
        plt.xlabel("向量数量")
        plt.ylabel(metric_names[metric])
        plt.xscale('log')
        # 对于召回率，使用线性刻度
        if metric != "recall_k1":
            plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # 保存图表
        output_dir = "tests/performance_charts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{dataset}_{metric}.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Saved chart: {output_file}")

# 生成综合对比图表
for max_vectors in max_vectors_list:
    plt.figure(figsize=(14, 10))
    
    # 为每个指标创建子图
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        
        # 为每个数据库创建数据点
        for db in databases:
            x = []
            y = []
            for dataset in datasets:
                if db in results[dataset][max_vectors]:
                    x.append(dataset)
                    # 根据指标类型获取正确的数据
                    if metric == "insert_throughput":
                        if "insert" in results[dataset][max_vectors][db] and "throughput" in results[dataset][max_vectors][db]["insert"]:
                            y.append(results[dataset][max_vectors][db]["insert"]["throughput"])
                        else:
                            y.append(0)
                    elif metric == "search_qps_k1":
                        if "search" in results[dataset][max_vectors][db] and "qps" in results[dataset][max_vectors][db]["search"]:
                            y.append(results[dataset][max_vectors][db]["search"]["qps"])
                        else:
                            y.append(0)
                    elif metric == "search_latency_k1":
                        if "search" in results[dataset][max_vectors][db] and "avg_latency" in results[dataset][max_vectors][db]["search"]:
                            y.append(results[dataset][max_vectors][db]["search"]["avg_latency"])
                        else:
                            y.append(0)
                    elif metric == "recall_k1":
                        if "search" in results[dataset][max_vectors][db] and "recall" in results[dataset][max_vectors][db]["search"]:
                            y.append(results[dataset][max_vectors][db]["search"]["recall"])
                        else:
                            y.append(0)
                    elif metric == "concurrent_qps_5":
                        if "concurrent_search" in results[dataset][max_vectors][db] and "qps" in results[dataset][max_vectors][db]["concurrent_search"]:
                            y.append(results[dataset][max_vectors][db]["concurrent_search"]["qps"])
                        else:
                            y.append(0)
                    else:
                        y.append(0)
            
            # 绘制柱状图
            plt.bar(np.arange(len(x)) + 0.2 * (databases.index(db) - 0.5), y, width=0.4, label=db)
        
        # 设置子图标题和标签
        plt.title(metric_names[metric])
        plt.xticks(np.arange(len(datasets)), datasets)
        # 对于召回率，使用线性刻度
        if metric != "recall_k1":
            plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_file = f"{output_dir}/综合对比_{max_vectors}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved chart: {output_file}")

print("图表生成完成！")