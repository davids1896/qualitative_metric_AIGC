import os
from scipy.linalg import sqrtm
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from skimage.transform import resize
from tqdm import tqdm  # 导入tqdm库

def load_images(folder_path, target_size=(299, 299)):
    images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 支持的图像格式
    image_files = os.listdir(folder_path)  # 获取文件列表
    # 获取文件夹中的所有图像文件
    for img_file in tqdm(image_files, desc="Loading images", unit="files"):
        if img_file.lower().endswith(supported_formats):  # 仅处理支持的格式
            img_path = os.path.join(folder_path, img_file)
            try:
                # 加载图像并调整大小
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                images.append(img_array)
            except UnidentifiedImageError:
                print(f"无法识别的图像文件: {img_path}")
            except Exception as e:
                print(f"加载图像时出错: {img_path}, 错误: {e}")
    return np.array(images)

# 指定数据集路径
dataset_baseline_path = '/vepfs-sha/yuming.li/shuffusion/prompt/laion5b/images_from_windows/laion_images/selected'
dataset_evaluated_path = '/vepfs-sha/yuming.li/shuffusion/images/ScaleCrafter/23_2048_2048'

# 加载图像
images1 = load_images(dataset_baseline_path)

print("Images loaded from dataset1:", len(images1))

images2 = load_images(dataset_evaluated_path)

print("Images loaded from dataset2:", len(images2))


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
    
    # 计算FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 加载预训练的InceptionV3模型
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

print("Model loaded")

# 预处理图像
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

print("Images preprocessed")

print("Computing FID")

# 计算FID
fid_value = calculate_fid(model, images1, images2)
print("FID:", fid_value)


# 示例：加载您的真实图像和生成图像
# images1, images2 = load_images(dataset1), load_images(dataset2)
# fid_value = calculate_fid(model, images1, images2)
# print("FID:", fid_value)
