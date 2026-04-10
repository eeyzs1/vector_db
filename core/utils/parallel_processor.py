import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Tuple

class ParallelProcessor:
    def __init__(self, max_workers: int = 4):
        """初始化并行处理器
        
        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
    
    def process_batch(self, items: List[Any], func: Callable[[Any], Any]) -> List[Any]:
        """批量处理任务
        
        Args:
            items: 待处理的项目列表
            func: 处理函数
            
        Returns:
            处理结果列表
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(func, item): item for item in items}
            
            # 收集结果
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"处理任务时出错: {e}")
                    results.append(None)
        
        return results
    
    def process_batch_with_args(self, items: List[Tuple[Any, ...]], func: Callable[..., Any]) -> List[Any]:
        """批量处理带参数的任务
        
        Args:
            items: 待处理的参数元组列表
            func: 处理函数
            
        Returns:
            处理结果列表
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(func, *item): item for item in items}
            
            # 收集结果
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"处理任务时出错: {e}")
                    results.append(None)
        
        return results

# 全局并行处理器实例
parallel_processor = ParallelProcessor()