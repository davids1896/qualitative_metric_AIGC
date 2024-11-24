import os
import torch
import clip
from PIL import Image

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
    # return similarity.item()

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 指定图像目录路径
image_directory = '/vepfs-sha/yuming.li/shuffusion/images/DemoFusion/23_1024_1024/selected'

# 获取所有图像的文件路径，并提取对应的文本
image_paths = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory)]
texts = [os.path.splitext(fname)[0] for fname in os.listdir(image_directory)]

# 计算每张图像与其对应文本的CLIP分数
total_score = 0.0
for image_path, text in zip(image_paths, texts):
    # 加载和预处理图像
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 将文本标记化并转移到设备
    text = clip.tokenize([text]).to(device)
    
    # 计算CLIP分数
    clip_score = calculate_clip_score(image, text, model, device)
    print(f"CLIP Score for {os.path.basename(image_path)}: {clip_score}")
    
    # 累加分数
    total_score += clip_score

# 计算平均CLIP分数
average_clip_score = total_score / len(image_paths)
print("Average CLIP Score:", average_clip_score)
