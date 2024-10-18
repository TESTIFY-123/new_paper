import os
import numpy as np
import cv2  # 用于读取png文件
from scipy.io import savemat
from utility.util import crop_center, Visualize3D, minmax_normalize
# 1. 主目录路径
main_folder_path = '/data/HSI_Data/cave'

# 2. 获取主目录下所有子文件夹的路径
subfolders = [os.path.join(main_folder_path, folder) for folder in os.listdir(main_folder_path) if
              os.path.isdir(os.path.join(main_folder_path, folder))]

# 3. 遍历每个子文件夹
for subfolder in subfolders:
    print(f"正在处理文件夹: {subfolder}")

    # 获取子文件夹中的所有 .png 文件
    png_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
    png_files.sort()  # 确保通道按照正确顺序排列

    # 读取并堆叠所有的通道
    channels = []
    for file in png_files:
        img_path = os.path.join(subfolder, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 读取灰度图
        channels.append(img)

    # 将通道合并为一个3D数组（高度，宽度，通道数）
    hsi_image = np.stack(channels, axis=-1)

    # 4. 最大最小归一化

    hsi_normalized = minmax_normalize(hsi_image)

    # 5. 保存为 .mat 文件
    # 为每个子文件夹生成一个唯一的文件名
    output_file_name = os.path.basename(subfolder) + '_normalized_hsi.mat'
    output_path = os.path.join(main_folder_path, output_file_name)

    savemat(output_path, {'hsi': hsi_normalized})

    print(f"HSI 图像已保存为 {output_path}")

print("所有文件夹处理完成！")