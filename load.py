# from datasets import load_dataset
#
#
# ds = load_dataset("danaroth/icvl",
#                   cache_dir='icvl',
#                   download_mode="force_redownload",
#                   streaming=False)
import h5py

# import os
# import shutil
# from sklearn.model_selection import train_test_split
#
# # 定义数据集所在文件夹路径
# data_dir = '/data/HSI_Data/Hyperspectral_Project/apex_crop/'  # 数据集文件夹路径
# train_dir = '/data/HSI_Data/Hyperspectral_Project/train'   # 训练集文件夹路径
# test_dir = '/data/HSI_Data/Hyperspectral_Project/test'  # 测试集文件夹路径
#
# # 创建存储训练集和测试集的文件夹
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
#
# # 列出数据文件夹中的所有文件
# file_list = os.listdir(data_dir)
# file_list = [f for f in file_list if os.path.isfile(os.path.join(data_dir, f))]  # 仅保留文件
#
# # 按 70% 训练集和 30% 测试集的比例拆分数据
# train_files, test_files = train_test_split(file_list, test_size=0.3, random_state=42)
#
# # 将训练集文件复制到训练集文件夹
# for file_name in train_files:
#     src = os.path.join(data_dir, file_name)
#     dst = os.path.join(train_dir, file_name)
#     shutil.copy(src, dst)  # 复制文件到训练集文件夹
#
# # 将测试集文件复制到测试集文件夹
# for file_name in test_files:
#     src = os.path.join(data_dir, file_name)
#     dst = os.path.join(test_dir, file_name)
#     shutil.copy(src, dst)  # 复制文件到测试集文件夹
#
# print(f"训练集文件数量: {len(train_files)}")
# print(f"测试集文件数量: {len(test_files)}")
