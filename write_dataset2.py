from datasets import load_dataset
import os
import json
from pathlib import Path
import shutil
import pandas as pd
from glob import glob
from io import BytesIO
from PIL import Image

# 创建保存数据的文件夹
save_dir = "/mnt/weka/yt_workspace/Lumina-Image-2.0/minidataset"
images_dir = os.path.join(save_dir, "images_pdd3")

# 删除已存在的文件夹（如果有）并重新创建
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# 创建一个JSON列表来存储数据
data_list = []

# 获取所有parquet文件
parquet_file_list = glob("/mnt/weka/yt_workspace/Lumina-Image-2.0/dataset/pdd3/*.parquet")

# 计数器用于图像命名
image_counter = 0

# 处理每个parquet文件
for parquet_file in parquet_file_list:
    print(f"正在处理文件: {parquet_file}")
    
    # 读取parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 处理每一行数据
    for _, row in df.iterrows():
        # 提取图像和描述
        image_bytes = row["img"]["bytes"]
        caption = row["caption"]
        
        # 保存图像
        image_filename = f"image_{image_counter:05d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        
        # 将二进制数据转换为图像并保存
        image = Image.open(BytesIO(image_bytes))
        image.save(image_path)
        
        # 创建数据项
        data_item = {
            "image_path": os.path.join("images_pdd3", image_filename),
            "prompt": caption
        }
        
        # 添加到数据列表
        data_list.append(data_item)
        
        # 更新计数器
        image_counter += 1
        
        # 打印进度
        if image_counter % 100 == 0:
            print(f"已处理 {image_counter} 个样本")

# 将数据列表保存为JSON文件
with open(os.path.join(save_dir, "data_pdd3.json"), "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"已成功保存 {len(data_list)} 个样本到 {save_dir} 文件夹")