import numpy as np
import scipy.io as sio
from utility.util import crop_center, Visualize3D, minmax_normalize
import os

# 读取原始的 .mat 文件
dst_folder='/data/HSI_Data/test_icvl_256/'
src_file='/data/HSI_Data/test_icvl_512'
for filename in os.listdir(src_file):
    mat_data = sio.loadmat(os.path.join(src_file,filename))

    # 假设矩阵存储在某个键中，比如 'data'，我们提取出这个矩阵
    # 并假设其形状为 (height, width, num_channels)，即 (h, w, C)
    matrix = mat_data['data']

    # 检查矩阵形状
    print("Original matrix shape:", matrix.shape)  # (h, w, num_channels)

    # 定义窗口滑动的参数：假设每次提取 31 个通道
    window_size = 31

    # 假设你希望滑动窗口开始于第 0 个通道，可以通过循环来滑动窗口
    # 可以根据需求选择起始点，比如从通道 0 开始，滑动一个窗口

    # 以步长为 1 滑动窗口，提取多个 31 个通道的片段
    step_size = 5
    for start_channel in range(0, matrix.shape[2] - window_size + 1, step_size):
        sliced_matrix = matrix[:, :, start_channel:start_channel + window_size]

        sliced_matrix = sliced_matrix.transpose((2, 1, 0))
        sliced_matrix = crop_center(sliced_matrix, 256, 256)
        sliced_matrix = minmax_normalize(sliced_matrix)
        sliced_matrix = sliced_matrix.transpose((2, 1, 0))

        # 为每个提取的片段生成一个唯一的文件名

        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        # 保存每个片段
        sio.savemat(dst_folder + filename, {'data': sliced_matrix})
        print(f"Sliced data from channel {start_channel} to {start_channel + window_size - 1} saved to '{filename}'",
              f"{np.shape(sliced_matrix)}")

# mat_data = sio.loadmat('/data/HSI_Data/resources/Pavia.mat')
#
# # 假设矩阵存储在某个键中，比如 'data'，我们提取出这个矩阵
# # 并假设其形状为 (height, width, num_channels)，即 (h, w, C)
# matrix = mat_data['pavia']
#
# # 检查矩阵形状
# print("Original matrix shape:", matrix.shape)  # (h, w, num_channels)
#
# # 定义窗口滑动的参数：假设每次提取 31 个通道
# window_size = 31
#
# # 假设你希望滑动窗口开始于第 0 个通道，可以通过循环来滑动窗口
# # 可以根据需求选择起始点，比如从通道 0 开始，滑动一个窗口
#
# # 以步长为 1 滑动窗口，提取多个 31 个通道的片段
# step_size = 5
# for start_channel in range(0, matrix.shape[2] - window_size + 1, step_size):
#     sliced_matrix = matrix[:, : , start_channel:start_channel + window_size]
#
#     sliced_matrix=sliced_matrix.transpose((2,1,0))
#     sliced_matrix=crop_center(sliced_matrix,256,256)
#     sliced_matrix=minmax_normalize(sliced_matrix)
#     sliced_matrix = sliced_matrix.transpose((2, 1, 0))
#
#
#     # 为每个提取的片段生成一个唯一的文件名
#     filename = f'sliced_pavia_data_{start_channel}.mat'
#
#     # 保存每个片段
#     sio.savemat(dst_folder+filename, {'data': sliced_matrix})
#     print(f"Sliced data from channel {start_channel} to {start_channel + window_size - 1} saved to '{filename}'",f"{np.shape(sliced_matrix)}")


