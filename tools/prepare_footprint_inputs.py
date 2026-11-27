import os
import argparse
import rasterio
from PIL import Image
import numpy as np
import json


def geotiff_to_png_and_coco_json(input_folder, output_folder, json_output_path, overwrite=True):
    """
    将三通道GeoTIFF转换为PNG，并将文件名记录到COCO格式的JSON文件中。
    
    参数:
    input_folder (str): 输入GeoTIFF文件夹路径
    output_folder (str): 输出PNG文件夹路径
    json_output_path (str): 输出COCO JSON文件路径
    """
    os.makedirs(output_folder, exist_ok=True)

    # 初始化COCO格式的字典
    coco_data = {
        "categories": [
            {
                "id": 1,
                "name": "building",
                "supercategory": "none"
            }
        ],
        "images": [],
    }

    # 图像ID计数器
    image_id = 0

    # 遍历输入文件夹中的所有文件
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith((".tif", ".tiff")):
            # 构建输入和输出文件路径
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)

            try:
                if os.path.exists(output_path) and not overwrite:
                    print(f"跳过 {filename}：PNG已存在 {output_path}")
                    continue

                # 读取GeoTIFF文件
                with rasterio.open(input_path) as src:
                    # 检查通道数和数据类型
                    if src.count != 3:
                        print(f"跳过 {filename}：不是三通道图像（通道数: {src.count}）")
                        continue
                    if src.dtypes[0] != 'uint8':
                        print(f"跳过 {filename}：数据类型不是uint8（类型: {src.dtypes[0]}）")
                        continue

                    # 读取RGB数据
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)

                    # 堆叠为RGB数组
                    rgb = np.stack([r, g, b], axis=-1)

                    # 创建并保存PNG
                    image = Image.fromarray(rgb, mode='RGB')
                    image.save(output_path, 'PNG')
                    print(f"已转换 {filename} 到 {output_path}")

                    # 添加到COCO images字段
                    coco_data["images"].append({
                        "file_name": output_filename,
                        "id": image_id,
                        "width": src.width,
                        "height": src.height,
                        "date_captured": "2025-03-13",  # 可根据需要调整
                        "license": 1  # 可选，假设默认许可
                    })
                    image_id += 1

            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    # 保存COCO JSON文件
    output_dir = os.path.dirname(json_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(json_output_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)
    print(f"COCO JSON文件已保存到: {json_output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF patches to PNG images and generate a COCO-style annotation JSON for MLS-BRN inference."
    )
    parser.add_argument("--input-folder", "--input_folder", dest="input_folder", required=True)
    parser.add_argument("--output-folder", "--output_folder", dest="output_folder", required=True)
    parser.add_argument("--json-output", "--json_output", dest="json_output_path", required=True)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing PNG files.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    geotiff_to_png_and_coco_json(
        args.input_folder,
        args.output_folder,
        args.json_output_path,
        overwrite=not args.no_overwrite,
    )
