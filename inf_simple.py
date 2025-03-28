import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
# from diffusers import Lumina2Pipeline
from diffusers.pipelines import Lumina2Text2ImgPipeline


from diffusers import StableDiffusionPipeline

pipe = Lumina2Text2ImgPipeline.from_pretrained("Alpha-VLLM/Lumina-Image-2.0", torch_dtype=torch.bfloat16)
pipe.to("cuda")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pth = torch.load("/mnt/weka/yt_workspace/Lumina-Image-2.0/results/NextDiT_2B_GQA_patch2_Adaln_Refiner_bs16_lr2e-4_bf16/checkpoints/0003000/consolidated.00-of-01.pth")
# import pdb;pdb.set_trace()
print("pth key:",pth.keys())
print("model key:", pipe.transformer.state_dict().keys())


def map_pth_key_to_model_key(pth_key):
    """
    推测并转换 pth key 到 model key.

    Args:
        pth_key: 原始的 pth key 字符串.

    Returns:
        对应的 model key 字符串，如果无法推测则返回 None 或原始 key (取决于需求).
    """

    key_mapping = {
        "x_embedder": "x_embedder",
        "t_embedder.mlp": "time_caption_embed.timestep_embedder",
        "cap_embedder": "time_caption_embed.caption_embedder",
        "noise_refiner": "noise_refiner",
        "context_refiner": "context_refiner",
        "layers": "layers",
        "norm_final": "norm_out.linear_1",
        "final_layer.linear": "norm_out.linear_2",
        # 注意: final_layer.adaLN_modulation 在 model key 中似乎没有直接对应，这里先忽略或根据实际情况处理
        # "final_layer.adaLN_modulation": "...", # 需要进一步分析如何映射
        "attention": "attn",
        "feed_forward": "feed_forward",
        "attention_norm1": "norm1.norm",
        "attention_norm2": "norm2",
        "ffn_norm1": "ffn_norm1",
        "ffn_norm2": "ffn_norm2",
        "adaLN_modulation.1": "norm1.linear", # 针对 adaLN_modulation.1 映射到 norm1.linear
    }

    parts = pth_key.split('.')
    model_key_parts = []

    module_name = parts[0]
    if module_name in key_mapping:
        model_key_parts.append(key_mapping[module_name])
        parts = parts[1:] # 移除已映射的 module_name
    else:
        return pth_key # 如果最外层模块名都无法映射，直接返回原 key 或根据需求处理


    for i in range(len(parts)):
        part = parts[i]
        if part == "noise_refiner" or part == "context_refiner" or part == "layers":
            if i + 1 < len(parts) and parts[i+1].isdigit(): # 处理索引 e.g., noise_refiner.0
                model_key_parts.append(part + "." + parts[i+1])
                parts = parts[i+2:] # 跳过已处理的 module 和 index 部分
                break # 继续处理剩余 parts
            else:
                model_key_parts.append(part)
                parts = parts[i+1:]
                break # 继续处理剩余 parts
        elif part == "mlp":
            continue # t_embedder.mlp 已经被整体映射了, mlp 部分不需要再处理
        elif part == "feed_forward":
            model_key_parts.append("feed_forward")
        elif part == "attention":
            model_key_parts.append("attn")
            if i + 1 < len(parts) and parts[i+1] == "qkv": # 特殊处理 attention.qkv.weight
                if i + 2 < len(parts) and parts[i+2] == "weight":
                    prefix = ".".join(model_key_parts) + ".to_"
                    return [prefix + qkv + ".weight" for qkv in ["q", "k", "v"]] # 返回 q, k, v 三个 key 的列表
                else:
                    model_key_parts.append("to_out.0") # attention.out 映射到 attn.to_out.0
                    parts = parts[i+1:] # 跳过 attention.out 部分
                    break
            elif i + 1 < len(parts) and parts[i+1] == "out":
                model_key_parts.append("to_out.0") # attention.out 映射到 attn.to_out.0
                parts = parts[i+1:] # 跳过 attention.out 部分
                break
            elif i + 1 < len(parts) and parts[i+1] == "q_norm":
                model_key_parts.append("norm_q")
                parts = parts[i+1:]
                break
            elif i + 1 < len(parts) and parts[i+1] == "k_norm":
                model_key_parts.append("norm_k")
                parts = parts[i+1:]
                break
            elif part == "attention_norm1":
                model_key_parts.append("norm1.norm")
                parts = parts[i+1:]
                break
            elif part == "attention_norm2":
                model_key_parts.append("norm2")
                parts = parts[i+1:]
                break
            else:
                model_key_parts.append("attn") # 兜底，如果只有 attention
        elif part == "attention_norm1":
            model_key_parts.append("norm1.norm")
        elif part == "attention_norm2":
            model_key_parts.append("norm2")
        elif part == "ffn_norm1":
            model_key_parts.append("ffn_norm1")
        elif part == "ffn_norm2":
            model_key_parts.append("ffn_norm2")
        elif part == "adaLN_modulation":
            if i + 1 < len(parts) and parts[i+1] == "1":
                model_key_parts.append("norm1.linear") #adaLN_modulation.1 映射到 norm1.linear
                parts = parts[i+2:] # 跳过 adaLN_modulation.1
                break
            else:
                model_key_parts.append("adaLN_modulation") # 兜底，虽然根据你的例子应该不会出现
        elif part == "norm_final":
             model_key_parts.append("norm_out.linear_1")
        elif part == "final_layer":
            if i + 1 < len(parts) and parts[i+1] == "linear":
                model_key_parts.append("norm_out.linear_2") # final_layer.linear 映射到 norm_out.linear_2
                parts = parts[i+2:]
                break
            elif part == "adaLN_modulation":
                pass # 忽略 final_layer.adaLN_modulation, 或根据实际情况处理
            else:
                model_key_parts.append("final_layer") # 兜底
        elif part.isdigit(): # 处理 layers.0.xxx 中的数字index
            model_key_parts.append(part)
        elif part == "weight" or part == "bias":
            model_key_parts.append(part)
        else:
            model_key_parts.append(part) # 其他部分直接添加

    if pth_key.endswith("attention.qkv.weight"): # 特殊处理 attention.qkv.weight 的情况，在循环外再次判断
        return None # 已经在循环内处理了，这里返回 None 避免重复处理或者根据需求返回其他

    if not model_key_parts: # 如果 model_key_parts 为空，说明没有成功映射，返回 None 或原始 key
        return pth_key # 或者 return None

    return ".".join(model_key_parts)


# 示例用法
pth_keys_odict = pth.keys() #['x_embedder.weight', 'x_embedder.bias', 'noise_refiner.0.attention.qkv.weight', 'noise_refiner.0.attention.out.weight', 'noise_refiner.0.attention_norm1.weight', 'noise_refiner.0.ffn_norm1.weight', 'layers.0.attention.qkv.weight', 'layers.0.feed_forward.w1.weight', 'norm_final.weight', 'final_layer.linear.weight', 'final_layer.adaLN_modulation.1.bias']

model_keys_list = []
from collections import OrderedDict
model_keys_odict = OrderedDict()
for pth_key in pth_keys_odict:
    mapped_key = map_pth_key_to_model_key(pth_key)
    model_keys_odict[mapped_key] = pth[pth_key]

# 将 OrderedDict 转换为普通字典
# model_keys_dict = dict(model_keys_odict)

# 打印模型参数数量
# print("Pth Keys:", pth_keys_odict)
# print("\nModel Keys 推测:")
# for model_key in model_keys_list:
#     print(model_key)

pipe.transformer.load_state_dict(model_keys_odict,strict=False)
# exit()
# pipe.transformer.load_state_dict(pth)
prompt = "A serene photograph capturing the golden reflection of the sun on a vast expanse of water. The sun is positioned at the top center, casting a brilliant, shimmering trail of light across the rippling surface. The water is textured with gentle waves, creating a rhythmic pattern that leads the eye towards the horizon. The entire scene is bathed in warm, golden hues, enhancing the tranquil and meditative atmosphere. High contrast, natural lighting, golden hour, photorealistic, expansive composition, reflective surface, peaceful, visually harmonious."
# prompt = "A realistic photo of a chinese girl, with the background is a beautiful landscape"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=20,
    # cfg_trunc_ratio=0.25,
    # cfg_normalization=True,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("lumina_demo_2.png")
