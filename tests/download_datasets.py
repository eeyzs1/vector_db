import os
import wget
import tarfile
import numpy as np
import requests

def download_file(url, save_path):
    """下载文件，支持HTTP和FTP链接"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading {url}...")
    
    try:
        # 尝试使用wget下载
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        wget.download(url, save_path)
        print(f"\nDownloaded successfully to {save_path}")
        return True
    except Exception as e:
        print(f"wget下载失败: {str(e)}")
        
        # 尝试使用requests下载，忽略SSL验证
        try:
            print("尝试使用requests下载...")
            response = requests.get(url, stream=True, timeout=300, verify=False)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Downloaded successfully to {save_path}")
            return True
        except Exception as e:
            print(f"requests下载失败: {str(e)}")
            return False

def extract_tar_gz(tar_path, extract_dir):
    """解压tar.gz文件"""
    print(f"Extracting {tar_path}...")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print(f"Extracted successfully to {extract_dir}")
        return True
    except Exception as e:
        print(f"解压失败: {str(e)}")
        return False

def download_and_extract_dataset(dataset_name, url, data_dir):
    """下载并解压数据集"""
    tar_filename = os.path.join(data_dir, f"{dataset_name}.tar.gz")
    extract_dir = os.path.join(data_dir, dataset_name)
    
    # 下载数据集
    if not os.path.exists(tar_filename):
        if not download_file(url, tar_filename):
            print(f"无法下载{dataset_name}数据集，请手动下载")
            print(f"下载链接: {url}")
            return False
    else:
        print(f"{tar_filename}已存在，跳过下载")
    
    # 解压数据集
    if not os.path.exists(extract_dir):
        if not extract_tar_gz(tar_filename, data_dir):
            print(f"无法解压{dataset_name}数据集")
            return False
    else:
        print(f"{extract_dir}已存在，跳过解压")
    
    return True

def load_sift1m(data_dir):
    """加载SIFT1M数据集"""
    # SIFT1M数据集文件
    base_file = os.path.join(data_dir, 'sift', 'sift_base.fvecs')
    query_file = os.path.join(data_dir, 'sift', 'sift_query.fvecs')
    groundtruth_file = os.path.join(data_dir, 'sift', 'sift_groundtruth.ivecs')
    
    if not (os.path.exists(base_file) and os.path.exists(query_file) and os.path.exists(groundtruth_file)):
        print("SIFT1M数据集文件不存在，请确保已正确下载和解压")
        return None, None, None
    
    def read_fvecs(file):
        """读取fvecs文件"""
        with open(file, 'rb') as f:
            while True:
                try:
                    dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                    yield vec
                except:
                    break
    
    def read_ivecs(file):
        """读取ivecs文件"""
        with open(file, 'rb') as f:
            while True:
                try:
                    dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
                    yield vec
                except:
                    break
    
    print("Loading SIFT1M dataset...")
    # 读取base向量（100万向量）
    base_vectors = list(read_fvecs(base_file))
    print(f"Loaded {len(base_vectors)} base vectors")
    
    # 读取query向量（1000向量）
    query_vectors = list(read_fvecs(query_file))
    print(f"Loaded {len(query_vectors)} query vectors")
    
    # 读取groundtruth
    groundtruth = list(read_ivecs(groundtruth_file))
    print(f"Loaded {len(groundtruth)} groundtruth vectors")
    
    return base_vectors, query_vectors, groundtruth

def load_gist1m(data_dir):
    """加载GIST1M数据集"""
    # GIST1M数据集文件
    base_file = os.path.join(data_dir, 'gist', 'gist_base.fvecs')
    query_file = os.path.join(data_dir, 'gist', 'gist_query.fvecs')
    groundtruth_file = os.path.join(data_dir, 'gist', 'gist_groundtruth.ivecs')
    
    if not (os.path.exists(base_file) and os.path.exists(query_file) and os.path.exists(groundtruth_file)):
        print("GIST1M数据集文件不存在，请确保已正确下载和解压")
        return None, None, None
    
    def read_fvecs(file):
        """读取fvecs文件"""
        with open(file, 'rb') as f:
            while True:
                try:
                    dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                    yield vec
                except:
                    break
    
    def read_ivecs(file):
        """读取ivecs文件"""
        with open(file, 'rb') as f:
            while True:
                try:
                    dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
                    yield vec
                except:
                    break
    
    print("Loading GIST1M dataset...")
    # 读取base向量（100万向量）
    base_vectors = list(read_fvecs(base_file))
    print(f"Loaded {len(base_vectors)} base vectors")
    
    # 读取query向量（1000向量）
    query_vectors = list(read_fvecs(query_file))
    print(f"Loaded {len(query_vectors)} query vectors")
    
    # 读取groundtruth
    groundtruth = list(read_ivecs(groundtruth_file))
    print(f"Loaded {len(groundtruth)} groundtruth vectors")
    
    return base_vectors, query_vectors, groundtruth

def main():
    """主函数"""
    # 数据目录
    data_dir = "data/datasets"
    
    # 数据集下载链接
    sift_url = "https://corpus-texmex.irisa.fr/texmex/corpus/sift.tar.gz"
    gist_url = "https://corpus-texmex.irisa.fr/texmex/corpus/gist.tar.gz"
    
    # 下载并解压SIFT1M
    print("\n=== 下载SIFT1M数据集 ===")
    if not download_and_extract_dataset("sift", sift_url, data_dir):
        print("SIFT1M数据集下载失败，请手动下载")
        print(f"下载链接: {sift_url}")
    
    # 下载并解压GIST1M
    print("\n=== 下载GIST1M数据集 ===")
    if not download_and_extract_dataset("gist", gist_url, data_dir):
        print("GIST1M数据集下载失败，请手动下载")
        print(f"下载链接: {gist_url}")
    
    # 加载SIFT1M数据集
    print("\n=== 加载SIFT1M数据集 ===")
    sift_base, sift_query, sift_groundtruth = load_sift1m(data_dir)
    
    # 加载GIST1M数据集
    print("\n=== 加载GIST1M数据集 ===")
    gist_base, gist_query, gist_groundtruth = load_gist1m(data_dir)
    
    if sift_base and gist_base:
        print("\nDataset loading completed!")
        print(f"SIFT1M: {len(sift_base)} base vectors, {len(sift_query)} query vectors")
        print(f"GIST1M: {len(gist_base)} base vectors, {len(gist_query)} query vectors")
    else:
        print("\n数据集加载失败，请确保已正确下载和解压")

if __name__ == "__main__":
    main()