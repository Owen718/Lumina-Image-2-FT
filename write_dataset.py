from datasets import load_dataset
import os
import json
from pathlib import Path
import shutil

# 创建保存数据的文件夹
save_dir = "/mnt/weka/yt_workspace/Lumina-Image-2.0/images_pdd"
images_dir = os.path.join(save_dir, "images_splash")

# # 删除已存在的文件夹（如果有）并重新创建
# if os.path.exists(save_dir):
#     shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# 创建一个JSON列表来存储数据
data_list = []

# 流式加载数据集
dataset = load_dataset("MeissonFlow/splash", num_proc=10) 

# 查看数据集的结构
print(dataset)
print("="*10)
print("#Number of train samples: ", len(dataset["train"]))
print("="*10)

# 访问训练集并保存数据
for i, example in enumerate(dataset["train"]):
    # 提取图像和描述
    image = example["img"]
    caption = example["caption"]
    
    # 保存图像
    image_filename = f"image_{i:05d}.jpg"
    image_path = os.path.join(images_dir, image_filename)
    # import pdb; pdb.set_trace()
    image.save(image_path)
    
    # 创建数据项
    data_item = {
        "image_path": os.path.join("images", image_filename),
        "prompt": caption
    }
    
    # 添加到数据列表
    data_list.append(data_item)
    
    # 打印进度
    if i % 100 == 0:
        print(f"已处理 {i} 个样本")
    
    # # 达到10000个样本后停止
    # if i >= 19999:
    #     break

# 将数据列表保存为JSON文件
with open(os.path.join(save_dir, "data.json"), "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"已成功保存 {len(data_list)} 个样本到 {save_dir} 文件夹")