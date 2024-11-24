import os
from scipy.linalg import sqrtm
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from skimage.transform import resize
from tqdm import tqdm  # 导入tqdm库
import json

def load_images(folder_path, target_size=(299, 299)):
    images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 支持的图像格式
    image_files = os.listdir(folder_path)  # 获取文件列表
    # 获取文件夹中的所有图像文件
    for img_file in tqdm(image_files, desc=f"Loading images from {folder_path}", unit="files"):
        if img_file.lower().endswith(supported_formats):  # 仅处理支持的格式
            img_path = os.path.join(folder_path, img_file)
            try:
                # 加载图像并调整大小
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                images.append(img_array)
            except Exception as e:
                print(f"加载图像时出错: {img_path}, 错误: {e}")
    return np.array(images)

def calculate_fid(model, images1, images2):
    # 将图像转化为适应Inception v3的大小
    images1 = np.array([resize(image, (299, 299, 3)) for image in images1])
    images2 = np.array([resize(image, (299, 299, 3)) for image in images2])
    
    # 计算图像的特征
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    # 计算均值和协方差
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # 计算平均差和协方差矩阵的根
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # 如果协方差矩阵是复数，则取其实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 固定的基线数据集路径
dataset_baseline_path = '/vepfs-sha/yuming.li/shuffusion/prompt/laion5b/images_from_windows/laion_images/selected'

# 加载预训练的InceptionV3模型
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
print("Model loaded")

# 读取需要处理的路径列表
path_file = '/vepfs-sha/davids1898/NSFusion/quantitative/path_need_evaluation_1114.txt'
with open(path_file, 'r') as f:
    paths = [line.strip() for line in f.readlines()]

# 加载基线数据集的图像并预处理
images1 = load_images(dataset_baseline_path)
images1 = preprocess_input(images1)
print(f"Images loaded and preprocessed from baseline dataset: {len(images1)}")

# 遍历每个路径并计算FID分数
for dataset_evaluated_path in paths:
    print(f"Evaluating FID for {dataset_evaluated_path}")
    
    # 加载要评估的数据集图像
    images2 = load_images(dataset_evaluated_path)
    images2 = preprocess_input(images2)
    
    print(f"Images loaded and preprocessed from evaluated dataset: {len(images2)}")
    
    # 计算FID分数
    fid_value = calculate_fid(model, images1, images2)
    print(f"FID for {dataset_evaluated_path}: {fid_value}")
    
    # 将结果存入字典
    result = {
        'path': dataset_evaluated_path,
        'fid_score': fid_value
    }

    # 生成基于路径的输出文件名
    sanitized_path = dataset_evaluated_path.replace('/', '_').replace(':', '')
    output_file = f'/vepfs-sha/davids1898/NSFusion/quantitative/FID/FID_result_{sanitized_path}.json'
    
    # 将结果保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {output_file}")
