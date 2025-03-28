import os
import json
import shutil
import pandas as pd
from glob import glob
from io import BytesIO
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

# 创建保存数据的文件夹
save_dir = "/mnt/weka/yt_workspace/Lumina-Image-2.0/images_pdd"
images_dir = os.path.join(save_dir, "images_pdd3")

# 删除已存在的文件夹（如果有）并重新创建
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# 处理单个图像的函数
def process_image(args):
    image_bytes, caption, image_counter, images_dir = args
    
    try:
        # 保存图像
        image_filename = f"image_{image_counter:05d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        
        # 将二进制数据转换为图像并保存
        image = Image.open(BytesIO(image_bytes))
        image.save(image_path, quality=95)  # 可以调整质量以加快速度
        print(f"已保存图像 {image_counter} 到 {image_path}")
        # 创建数据项
        data_item = {
            "image_path": os.path.join("images_pdd3", image_filename),
            "prompt": caption
        }
        
        return data_item
    except Exception as e:
        print(f"处理图像 {image_counter} 时出错: {e}")
        return None

# 主函数
def main():
    # 获取所有parquet文件
    parquet_file_list = glob("/mnt/weka/yt_workspace/Lumina-Image-2.0/dataset/pdd3/*.parquet")
    
    # 创建一个JSON列表来存储数据
    data_list = []
    
    # 计数器用于图像命名
    image_counter = 0
    
    # 创建进程池
    num_cores = 24 #mp.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行处理")
    
    for parquet_file in parquet_file_list:
        print(f"正在处理文件: {parquet_file}")
        
        # 读取parquet文件
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"处理文件 {parquet_file} 时出错: {e}")
            continue
        
        # 准备批处理参数
        args_list = []
        for _, row in df.iterrows():
            args = (row["img"]["bytes"], row["caption"], image_counter, images_dir)
            args_list.append(args)
            image_counter += 1
        
        # 使用多进程并行处理图像
        try:
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(pool.imap(process_image, args_list), total=len(args_list), 
                                desc="处理图像", unit="张"))
        except Exception as e:
            break
            
        
        # 添加有效结果到数据列表
        data_list.extend([r for r in results if r is not None])
        
        print(f"已处理文件: {parquet_file}, 当前总数: {len(data_list)}")
    
    # 将数据列表保存为JSON文件
    with open(os.path.join(save_dir, "data_pdd3.json"), "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"已成功保存 {len(data_list)} 个样本到 {save_dir} 文件夹")

if __name__ == "__main__":
    main()