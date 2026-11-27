import numpy as np
from rasterio.transform import from_origin
from scipy.special import softmax
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
import shutil

from .patch_positions import mosaic_origin, parse_patch_position, patch_position_bounds
from .raster_io import read_geotiff as read_raster, save_geotiff


def list_files_in_directory(directory):
    try:
        # 获取指定目录下的所有文件名
        filenames = [
            os.path.join(directory, name) for name in sorted(os.listdir(directory))
            if os.path.isfile(os.path.join(directory, name))
        ]
        return filenames
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")
        return []

def read_geotiff(file_path):
    return read_raster(file_path, band=1)

def _confidence_mosaic(patches, uncertainties, positions, shape, block_size=256):
    """Evaluate the original per-pixel softmax formula in bounded NumPy blocks.

    Patch boundaries partition the image into rectangles with the same ordered
    contributors. Stacking contributors on the last axis preserves the original
    reduction order, including irregular edge overlaps; uncovered pixels stay zero.
    Temporary stacks are bounded by block_size rather than full image dimensions.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not (len(patches) == len(uncertainties) == len(positions)):
        raise ValueError("Each height patch requires one uncertainty patch and position")
    height, width = shape
    result = np.zeros(shape, dtype=np.float64)
    rectangles = []
    x_edges, y_edges = {0, width}, {0, height}
    for index, (patch, uncertainty, (x, y)) in enumerate(zip(patches, uncertainties, positions)):
        if patch.ndim != 2 or uncertainty.shape != patch.shape:
            raise ValueError("Height and uncertainty patches must have the same 2D shape")
        bottom, right = y + patch.shape[0], x + patch.shape[1]
        if x < 0 or y < 0 or right > width or bottom > height:
            raise ValueError("Patch lies outside the mosaic extent")
        rectangles.append((index, x, y, right, bottom))
        x_edges.update((x, right))
        y_edges.update((y, bottom))
    x_edges, y_edges = sorted(x_edges), sorted(y_edges)
    for top, bottom in zip(y_edges, y_edges[1:]):
        active = [rect for rect in rectangles if rect[2] <= top and rect[4] >= bottom]
        if not active:
            continue
        for left, right in zip(x_edges, x_edges[1:]):
            contributors = [rect for rect in active if rect[1] <= left and rect[3] >= right]
            if not contributors:
                continue
            for row in range(top, bottom, block_size):
                row_end = min(row + block_size, bottom)
                for col in range(left, right, block_size):
                    col_end = min(col + block_size, right)
                    slices = [(idx, slice(row-y, row_end-y), slice(col-x, col_end-x))
                              for idx, x, y, _, _ in contributors]
                    image_stack = np.stack([patches[idx][ys, xs] for idx, ys, xs in slices], axis=-1)
                    uncertainty_stack = np.stack([uncertainties[idx][ys, xs] for idx, ys, xs in slices], axis=-1)
                    weights = 1 - uncertainty_stack / (np.sum(uncertainty_stack, axis=-1, keepdims=True) + 1e-5)
                    weights = softmax(weights, axis=-1)
                    result[row:row_end, col:col_end] = np.sum(weights * image_stack, axis=-1)
    return result


def _mean_mosaic(patches, positions, shape, block_size=256):
    """Average ordered contributors in bounded blocks without a full count map.

    The same patch-boundary partition as confidence fusion makes the contributor
    count constant per rectangle. Float64 additions retain the original patch
    order rather than using a float32 or pairwise reduction.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(patches) != len(positions):
        raise ValueError("Each height patch requires one position")
    height, width = shape
    result = np.zeros(shape, dtype=np.float64)
    rectangles = []
    x_edges, y_edges = {0, width}, {0, height}
    for index, (patch, (x, y)) in enumerate(zip(patches, positions)):
        if patch.ndim != 2:
            raise ValueError("Height patches must be 2D")
        bottom, right = y + patch.shape[0], x + patch.shape[1]
        if x < 0 or y < 0 or right > width or bottom > height:
            raise ValueError("Patch lies outside the mosaic extent")
        rectangles.append((index, x, y, right, bottom))
        x_edges.update((x, right))
        y_edges.update((y, bottom))
    x_edges, y_edges = sorted(x_edges), sorted(y_edges)
    for top, bottom in zip(y_edges, y_edges[1:]):
        active = [rect for rect in rectangles if rect[2] <= top and rect[4] >= bottom]
        for left, right in zip(x_edges, x_edges[1:]):
            contributors = [rect for rect in active if rect[1] <= left and rect[3] >= right]
            if not contributors:
                continue
            for row in range(top, bottom, block_size):
                row_end = min(row + block_size, bottom)
                for col in range(left, right, block_size):
                    col_end = min(col + block_size, right)
                    block = result[row:row_end, col:col_end]
                    for index, x, y, _, _ in contributors:
                        block += patches[index][row-y:row_end-y, col-x:col_end-x]
                    block /= len(contributors)
    return result


def mosaic_patches(patches, patch_meta, img_dir_path, window_size, output_dir, dir_tag, 
                  return_output=True, show_grayscale=True, use_conf=True, save_tif=True):
    
    output_dir = os.path.join(output_dir, dir_tag)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    meta = copy.deepcopy(patch_meta) # 仅仅是在保存时让tif存为int16，如果直接调用原来的meta会导致后续处理类型不一致的情况
    img_range = patch_position_bounds(img_dir_path)
    position = [parse_patch_position(img_path) for img_path in img_dir_path] # 根据文件名获取图像

    if isinstance(window_size, int):
        h, w = window_size, window_size
    else:
        h, w = window_size

    full_frame = np.zeros([img_range['max_y'] - img_range['min_y'] + h, img_range['max_x'] - img_range['min_x'] + w])
    
    if use_conf: # 根据置信度加权融合
        conf_dir_path = []
        for depth_path in img_dir_path:
            dir_path, file_name = os.path.split(depth_path)
            conf_path = os.path.join(os.path.split(dir_path)[0], 'uncertainty', file_name.replace('depth', 'uncertainty'))
            conf_dir_path.append(conf_path)
        conf = [read_geotiff(conf_path)[0] for conf_path in conf_dir_path]

        for k in range(len(patches)):
            meta[k].update({'count': 3, 'dtype': 'int16'})
            output_file = os.path.join(output_dir, os.path.basename(img_dir_path[k]))
            if show_grayscale:
                patch = np.repeat(patches[k][..., np.newaxis], 3, axis=-1)
            else:
                cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                patch = (cmap(patches[k].astype(np.int16))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            if save_tif:
                save_geotiff(output_file, patch.transpose(2, 0, 1), meta[k])

        relative_positions = [(x - img_range['min_x'], y - img_range['min_y']) for x, y in position]
        full_frame = _confidence_mosaic(patches, conf, relative_positions, full_frame.shape)

    else:
        for k in range(len(patches)):
            meta[k].update({'count': 3, 'dtype': 'int16'})
            output_file = os.path.join(output_dir, os.path.basename(img_dir_path[k]))

            if show_grayscale:
                patch = np.repeat(patches[k][..., np.newaxis], 3, axis=-1)
            else:
                cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                patch = (cmap(patches[k].astype(np.int16))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            if save_tif:
                save_geotiff(output_file, patch.transpose(2, 0, 1), meta[k])

        relative_positions = [(x - img_range['min_x'], y - img_range['min_y']) for x, y in position]
        full_frame = _mean_mosaic(patches, relative_positions, full_frame.shape)

    
    # 更新元数据以适应新的尺寸
    origin_pos = mosaic_origin(meta)
    full_meta = meta[0].copy()
    ul_x, ul_y, res_x, res_y = origin_pos['origin_x'], origin_pos['origin_y'], full_meta['transform'].a, -full_meta['transform'].e
    full_meta_transform = from_origin(ul_x, ul_y, res_x, res_y)
    full_meta.update({
        'count': 3,
        'height': full_frame.shape[0],
        'width': full_frame.shape[1],
        'dtype': 'int16',
        'transform': full_meta_transform
    })

    output_file = os.path.join(output_dir, f'output.tif')
    if show_grayscale:
        # full_frame = np.stack([full_frame] * 3, axis=0)
        full_frame = np.repeat(full_frame[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        full_frame = (cmap(full_frame.astype(np.int16))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
    if save_tif:
        save_geotiff(output_file, full_frame.transpose(2,0,1), full_meta) if save_tif else None

    return full_frame[:,:,0], full_meta if return_output else None


def partition_blocks(input_dir, output_dir, window_size=1024, overlap=0.5, block_size=None,
                  use_conf=True):
    # 收集所有图像信息
    image_info = []
    img_dir_path = list_files_in_directory(input_dir)
    for filename in img_dir_path:
        filename = os.path.basename(filename)
        x, y = parse_patch_position(filename)
        image_info.append({"filename": filename, "x": x, "y": y})

    if not image_info:
        raise ValueError("No valid images found in the input directory.")
    
    # 确定图像网格范围
    img_range = patch_position_bounds(img_dir_path)
    min_x, max_x, min_y, max_y = img_range['min_x'], img_range['max_x'], img_range['min_y'], img_range['max_y']

    # 计算 n*n 集合的网格, 每个 n*step 覆盖 n*step 像素
    step = int(window_size * (1 - overlap))
    n, m = tuple(int(x) for x in block_size.split(','))  # 将str的blocksize解析为int
    x_set_step = n * step  # 集合覆盖的范围，例如 n=3，step=512，则为 1536
    y_set_step = m * step  # 集合覆盖的范围，例如 n=3，step=512，则为 1536

    # 生成 x_sets 和 y_sets，考虑边界重叠
    x_sets = []
    y_sets = []

    # 正常集合（无重叠）
    current_x = min_x
    while current_x <= max_x - x_set_step + step:
        x_sets.append(current_x)
        current_x += x_set_step

    # x方向的边界回溯处理（允许重叠）
    if current_x <= max_x:
        x_sets.append(max(min_x, max_x - x_set_step + step))

    # 正常集合（无重叠）
    current_y = min_y
    while current_y <= max_y - y_set_step + step:
        y_sets.append(current_y)
        current_y += y_set_step

    # y方向的边界回溯处理（允许重叠）
    if current_y <= max_y:
        y_sets.append(max(min_y, max_y - y_set_step + step))

    # 为每个 n*n 集合分配图像
    set_id = 0
    for x_set in x_sets:
        for y_set in y_sets:
            set_images = []
            for i in range(n):
                for j in range(m):
                    x = x_set + i * step
                    y = y_set + j * step
                    for info in image_info:
                        if info["x"] == x and info["y"] == y:
                            set_images.append(info["filename"])

            # 处理不完整集合，尝试补充相邻图像
            if len(set_images) < n * m:
                print(f"Incomplete set at x_set={x_set}, y_set={y_set} with {len(set_images)} images, attempting to fill gaps.")
                
                # 需要补充图像的个数
                remaining_images = n * m - len(set_images)

                # 尝试在相邻的 x 和 y 方向上查找补充图像
                # 检查相邻的 x 和 y 位置（例如上一个或下一个位置的图像）
                for delta_x in [-step, step]:  # 向左（-step）或向右（+step）查找
                    for delta_y in [-step, step]:  # 向上（-step）或向下（+step）查找
                        if remaining_images <= 0:
                            break
                        
                        new_x_set = x_set + delta_x
                        new_y_set = y_set + delta_y
                        
                        # 确保新位置在图像范围内
                        if min_x <= new_x_set <= max_x and min_y <= new_y_set <= max_y:
                            # 查找新位置的图像
                            for i in range(n):
                                for j in range(m):
                                    x = new_x_set + i * step
                                    y = new_y_set + j * step
                                    for info in image_info:
                                        if info["x"] == x and info["y"] == y:
                                            if info["filename"] not in set_images:
                                                set_images.append(info["filename"])
                                                remaining_images -= 1
                                                if remaining_images <= 0:
                                                    break

            # 如果集合图像数量满足 n * m，保存集合
            if len(set_images) == n * m:
                set_dir = os.path.join(output_dir, f"tmp_set_{set_id}")
                os.makedirs(set_dir, exist_ok=True)
                os.makedirs(os.path.join(set_dir, 'depth'), exist_ok=True)
                if use_conf:
                    os.makedirs(os.path.join(set_dir, 'uncertainty'), exist_ok=True)
                for filename in set_images:
                    src_depth_path = os.path.join(input_dir, filename)
                    dst_depth_path = os.path.join(set_dir, 'depth', filename)
                    shutil.copy(src_depth_path, dst_depth_path)

                    if use_conf:
                        src_conf_dir = os.path.join(os.path.dirname(os.path.split(src_depth_path)[0]), 'uncertainty')
                        src_conf_path = os.path.join(src_conf_dir, filename.replace('depth', 'uncertainty'))
                        dst_conf_path = os.path.join(set_dir, 'uncertainty', filename.replace('depth', 'uncertainty'))
                        shutil.copy(src_conf_path, dst_conf_path)

                print(f"Saved {len(set_images)} images to {set_dir}")
                set_id += 1
            else:
                print(f"Skipped incomplete set at x_set={x_set}, y_set={y_set} with {len(set_images)} images")

    print(f"Total {set_id} sets created.")
