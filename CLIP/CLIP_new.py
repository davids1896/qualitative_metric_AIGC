import os
import torch
import clip
from PIL import Image, UnidentifiedImageError
import json

def calculate_clip_score(image, text, model, device):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 归一化特征向量
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    similarity = (image_features @ text_features.T).squeeze()
    scaled_similarity = similarity.item() * 100  # 乘以100放大
    return scaled_similarity

# 加载CLIP模型
device = "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 读取需要处理的路径列表
path_file = '/share/DavidHong/code/quantitative/path_need_evaluation_1114.txt'
with open(path_file, 'r') as f:
    paths = [line.strip() for line in f.readlines()]

# 文件记录路径
truncated_images_file = '/share/DavidHong/code/quantitative/CLIP/truncated_images.txt'

# 遍历每个路径并计算CLIP分数
for image_directory in paths:
    image_paths = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory)]
    texts = [os.path.splitext(fname)[0] for fname in os.listdir(image_directory)]

    total_score = 0.0
    valid_image_count = 0

    for image_path, text in zip(image_paths, texts):
        try:
            # 加载和预处理图像
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            
            # 将文本标记化并转移到设备
            text = clip.tokenize([text]).to(device)
            
            # 计算CLIP分数
            clip_score = calculate_clip_score(image, text, model, device)
            print(f"CLIP Score for {os.path.basename(image_path)}: {clip_score}")
            
            # 累加分数
            total_score += clip_score
            valid_image_count += 1

        except (OSError, UnidentifiedImageError) as e:
            # 如果图片无法处理，则记录文件路径并跳过
            print(f"Error processing {image_path}: {e}")
            with open(truncated_images_file, 'a') as f:
                f.write(f"{image_path}\n")
            continue

    # 如果有有效的图像，计算平均分数
    if valid_image_count > 0:
        average_clip_score = total_score / valid_image_count
        print(f"Average CLIP Score for {image_directory}: {average_clip_score}")

        # 将结果存入字典
        result = {
            'path': image_directory,
            'average_clip_score': average_clip_score
        }

        # 生成基于路径的输出文件名
        sanitized_path = image_directory.replace('/', '_').replace(':', '')
        output_file = f'/share/DavidHong/code/quantitative/CLIP/clip_result_{sanitized_path}.json'
        
        # 将结果保存到JSON文件
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Results saved to {output_file}")
    else:
        print(f"No valid images found in {image_directory}, skipping average CLIP score calculation.")
