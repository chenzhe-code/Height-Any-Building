import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from .raster_io import read_geotiff, save_geotiff

def ground_filter(img, output_dir, window_size=150, down_ratio=1, ground_ratio=0.1,
                  save_result=False, dir_tag='dem&dsm', img_name=''):
    output_dir = os.path.join(output_dir, dir_tag)
    if (not os.path.exists(output_dir)) & save_result:
        os.makedirs(os.path.join(output_dir, 'dem_sparse'))
        os.makedirs(os.path.join(output_dir, 'ndsm'))
    # 如果imgs是路径则读取，不是则直接赋值
    image, meta = img if isinstance(img, tuple) else read_geotiff(img, band=1)
    # image, meta = read_geotiff(img) if os.path.isfile(img) else img
    # image = image[:,:,0] if image.shape[2]==3 else None

    height, width = image.shape
    process_height = height // down_ratio
    process_width = width // down_ratio
    dem_sparse = np.empty([process_height, process_width])

    # 减少图像大小加速DEM算法处理
    process_image = cv2.resize(image, (process_width, process_height), interpolation=cv2.INTER_AREA)
    window_size = window_size//down_ratio
    window_len = int((window_size-1)/2)

    start_row = window_len
    end_row = int(process_height-window_len)
    start_col = window_len
    end_col = int(process_width-window_len)

    num_ground_point = int(ground_ratio * window_size**2)
    num_ground_point = max(1, num_ground_point)

    if num_ground_point == 1:
        kernel_size = 2 * window_len + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dem_sparse = cv2.erode(process_image, kernel, borderType=cv2.BORDER_REPLICATE)
    else:
        # Do not process the edges to reduce the time required for judgment
        # for i in tqdm(range(start_row, end_row), desc= "inside dem"):
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                window_subset = process_image[i-window_len:i+window_len+1, j-window_len:j+window_len+1]
                window_min_group = np.sort(window_subset.flatten())[0:num_ground_point]

                # Remove outliers
                group_mean = np.mean(window_min_group)
                group_std = np.std(window_min_group)
                window_min_final = []
                for q in range(len(window_min_group)):
                    temp = group_mean - window_min_group[q] # Only the minimum is detected
                    if temp <= 3 * group_std:
                        window_min_final.append(window_min_group[q])
                window_min = np.mean(window_min_final)

                # window_min = np.mean(sorted(window_subset.flatten())[0:100])

                dem_sparse[i][j] = window_min

        # Process edges
        for i in range(start_row, end_row):
            for j in range(window_len):
                window_subset = process_image[i-window_len:i+window_len+1, 0:window_len+j+1]
                window_subset_min = np.min(window_subset.flatten())
                dem_sparse[i][j] = window_subset_min

                window_subset = process_image[i-window_len:i+window_len+1, process_width-1-window_len-j:process_width]
                window_subset_min = np.min(window_subset.flatten())
                dem_sparse[i][process_width-1-j] = window_subset_min

        for i in range(start_col, end_col):
            for j in range(window_len):
                window_subset = process_image[0:window_len+j+1, i-window_len:i+window_len+1]
                window_subset_min = np.min(window_subset.flatten())
                dem_sparse[j][i] = window_subset_min

                window_subset = process_image[process_height-1-window_len-j:process_height, i-window_len:i+window_len+1]
                window_subset_min = np.min(window_subset.flatten())
                dem_sparse[process_height-1-j][i] = window_subset_min

        # Process corners
        for i in range(window_len):
            for j in range(window_len):
                dem_sparse[i][j] = np.min(process_image[0:window_len+i+1, 0:window_len+j+1].flatten())
                dem_sparse[i][process_width-1-j] = np.min(process_image[0:window_len+i+1, process_width-1-window_len-j:process_width].flatten())
                dem_sparse[process_height-1-i][j] = np.min(process_image[process_height-1-window_len-j:process_height:i+window_len+1, 0:window_len+j+1].flatten())
                dem_sparse[process_height-1-i][process_width-1-j] = np.min(process_image[process_height-1-window_len-j:process_height:i+window_len+1,
                                                                             process_width-1-window_len-j:process_width].flatten())

    # dem_sparse[0][0] = np.min(image[0:window_len, 0:window_len].flatten())
    # dem_sparse[0][width-1] = np.min(image[0:window_len, width-1-window_len:width-1].flatten())
    # dem_sparse[height-1][0] = np.min(image[height-1-window_len:height-1, 0:window_len].flatten())
    # dem_sparse[height-1][width-1] = np.min(image[height-1-window_len:height-1, width-1-window_len:width-1].flatten())

    dem_sparse = cv2.resize(dem_sparse, (width, height), interpolation=cv2.INTER_LINEAR)

    use_kernel = 0
    if use_kernel == 1:
        # Gaussian_Kernel = cv2.getGaussianKernel(window_size, sigma = 1) * cv2.getGaussianKernel(window_size, sigma = 1).T
        kernel_size = int(window_size / 5) + 1
        kernel = (kernel_size, kernel_size)
        dem_dense = cv2.GaussianBlur(dem_sparse, kernel, 1, 1)

    if use_kernel==1:
        ndsm = (image - dem_dense)
    elif use_kernel==0:
        ndsm = (image -dem_sparse)

    if save_result:
        # meta.update({
        # 'dtype': 'int8',
        # })
        dem_sparse_out = np.repeat(dem_sparse[..., np.newaxis], 3, axis=-1)
        ndsm_out = np.repeat(ndsm[..., np.newaxis], 3, axis=-1)
        # write_tif(fn_nDSM1_out, nDSM1_out, tif['transform'], tif['projection'])
        dem_output_path = os.path.join(output_dir, 'dem_sparse', img_name)
        ndsm_output_path = os.path.join(output_dir, 'ndsm', img_name)
        save_geotiff(dem_output_path, dem_sparse_out.transpose(2,0,1), meta)
        save_geotiff(ndsm_output_path , ndsm_out.transpose(2,0,1), meta)

    # print('ok')

    return {"ndsm": (ndsm, meta),
            "dem": (dem_sparse, meta)}
