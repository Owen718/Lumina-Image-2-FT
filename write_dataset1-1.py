from datasets import load_dataset
import os
import json
from pathlib import Path
import shutil
import concurrent.futures
import threading
from tqdm import tqdm

# 创建一个锁，用于安全地更新共享数据结构
data_lock = threading.Lock()

def process_batch(batch_data, start_idx, images_dir):
    """处理一批数据"""
    local_data_list = []
    
    for i, example in enumerate(batch_data):
        # 计算全局索引
        idx = start_idx + i
        
        # 提取图像和描述
        image = example["img"]
        caption = example["caption"]
        
        # 保存图像
        image_filename = f"image_{idx:05d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path)
        
        # 创建数据项
        data_item = {
            "image_path": os.path.join("images", image_filename),
            "prompt": caption
        }
        
        # 添加到本地数据列表
        local_data_list.append(data_item)
    
    return local_data_list

# 创建保存数据的文件夹
save_dir = "/mnt/weka/yt_workspace/Lumina-Image-2.0/images_pdd"
images_dir = os.path.join(save_dir, "images_splash")


os.makedirs(save_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# 全局数据列表
data_list = []

# 流式加载数据集
dataset = load_dataset("/mnt/weka/yt_workspace/Lumina-Image-2.0/dataset/splash", num_proc=64)

# 查看数据集的结构
print(dataset)
print("="*10)
print("#Number of train samples: ", len(dataset["train"]))
print("="*10)

# 参数设置
batch_size = 100  # 每个批次处理的样本数
num_workers = 64   # 工作线程数

# 计算总批次数
total_samples = len(dataset["train"])
num_batches = (total_samples + batch_size - 1) // batch_size

# 使用线程池并行处理数据
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    
    # 提交任务
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_data = dataset["train"][start_idx:end_idx]
        
        future = executor.submit(process_batch, batch_data, start_idx, images_dir)
        futures.append(future)
    
    # 使用tqdm显示进度
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理数据批次"):
        batch_results = future.result()
        # 使用锁安全地更新全局数据列表
        with data_lock:
            data_list.extend(batch_results)

# 将数据列表保存为JSON文件
with open(os.path.join(save_dir, "data_splash.json"), "w", encoding="utf-8") as f:
    json.dump(data_list, ensure_ascii=False, indent=2)

print(f"已成功保存 {len(data_list)} 个样本到 {save_dir} 文件夹")