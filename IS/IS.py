import os
import numpy as np
from scipy.stats import entropy
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from tqdm import tqdm  # 导入tqdm库

def inception_score(images, model, n_split=10, eps=1E-16):
    # 处理图像以适应Inception v3
    processed_images = np.array([resize(img, (299, 299, 3)) for img in images])

    # 使用模型预测图像
    preds = model.predict(processed_images)
    
    # 计算每个图像的熵
    conditional_entropy = -np.sum(preds * np.log(preds + eps), axis=1)
    
    # 计算边缘分布
    marginal_entropy = -np.sum(np.mean(preds, axis=0) * np.log(np.mean(preds, axis=0) + eps))
    
    # 计算Inception Score
    is_score = np.exp(marginal_entropy - np.mean(conditional_entropy))
    return is_score

def load_images(folder_path):
    images = []
    image_files = os.listdir(folder_path)  # 获取文件列表
    # 遍历文件夹中的所有文件
    for img_file in tqdm(image_files, desc="Loading images", unit="files"):
        # 完整的文件路径
        img_path = os.path.join(folder_path, img_file)
        # 加载图像，预处理图像尺寸
        img = load_img(img_path, target_size=(299, 299))
        # 将图像转换为数组
        img_array = img_to_array(img)
        # 加入到列表中
        images.append(img_array)
    return np.array(images)

# 指定您的图像存储路径
dataset_path = '/vepfs-sha/yuming.li/shuffusion/images/DemoFusion/23_1024_1024'

print("Loading images from", dataset_path)

# 调用函数加载图像
images = load_images(dataset_path)

print("Images loaded:", len(images))

# 加载预训练的InceptionV3模型
model = InceptionV3(include_top=True, weights='imagenet')

print("Model loaded")

images = preprocess_input(images)

print("Computing Inception Score")

is_value = inception_score(images, model)

print("Inception Score:", is_value)

# 示例：加载您的生成图像
# images = load_images(dataset)
# is_value = inception_score(images, model)
# print("Inception Score:", is_value)
