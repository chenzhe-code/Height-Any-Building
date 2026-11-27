import os
import os.path as osp

# The pip rasterio wheel bundles a matching PROJ database. Conda may inject an
# older PROJ_DATA path that is incompatible with that wheel.
os.environ.pop('PROJ_DATA', None)

import numpy as np
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

from models.dpt import DPT_DINOv2
import argparse
from tqdm import tqdm
import rasterio
from utils import geo_func
from glob import glob
import cv2

from img_aug.build_aug import AugTransform

def save_tiff_with_geo_info(input_path, output_path, data):
    """
    Saves the data as a TIFF file with the geographic information from the input TIFF.
    """
    if len(data.shape) == 2:  # Single channel image
        output = data.astype(np.float32, copy=False)[np.newaxis, ...]
    else:  # Multi-channel image
        output = np.moveaxis((data * 255).astype(np.uint8), -1, 0)

    with rasterio.open(input_path) as source:
        profile = source.profile.copy()
    profile.update(
        driver='GTiff',
        height=output.shape[1],
        width=output.shape[2],
        count=output.shape[0],
        dtype=output.dtype,
    )
    with rasterio.open(output_path, 'w', **profile) as destination:
        destination.write(output)

def get_arguments():
    parser = argparse.ArgumentParser(description="Inference of Height Estimation")
    parser.add_argument('--img_path', type=str, required=True, help="Input directory containing images.")
    parser.add_argument("--arch_name", type=str, default="vitl", help="Select a method for inference.")
    parser.add_argument("--mode", type=str, default="regression", help="Select a method for inference.")
    parser.add_argument("--checkpoint", type=str, default="./pretrain/RS3DAda_vitl_DPT_height.pth", help="Where restore model parameters from.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--processing_res", type=int, default=1022, help="Maximum resolution of processing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument('--get_conf', dest='get_conf', default=False, action='store_true', help='get the pred conf by std')
    parser.add_argument('--grayscale', dest='grayscale', default=False, action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_tif', default=False, dest='save_tif', action='store_true', help='save as the geotiff format')
    return parser.parse_args()

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def main():
    args = get_arguments()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 统一建立输出文件夹规范
    check_directory(osp.join(args.output_dir, 'depth'))
    if args.get_conf:
        check_directory(osp.join(args.output_dir, 'uncertainty'))
    
    head_configs = [
        {'name': 'regression', 'nclass': 1},
        {'name': 'segmentation', 'nclass': 8}
    ]
    model = DPT_DINOv2(encoder='vitl', head_configs=head_configs, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model.cuda()

    patch_transformed = Compose([
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
        ToTensorV2()
    ])

    aug_trans = AugTransform() if args.get_conf else None

    # 遍历文件夹下所有图片
    validation_images = glob(osp.join(args.img_path, '*'))
    
    for image_path in tqdm(validation_images, desc="Processing Images"):
        # 读取并处理成全图矩阵
        img0 = geo_func.load_multiband(image_path)
        if img0.shape[2] > 3:
            img0 = img0[:,:,[0,1,2]]
            
        img = img0.copy().astype(np.uint8)
        
        # 【新增】：计算长宽距离 14 的倍数还差多少像素
        orig_h, orig_w = img.shape[:2]
        pad_h = (14 - orig_h % 14) % 14
        pad_w = (14 - orig_w % 14) % 14
        
        # 【新增】：如果有余数，则使用 'reflect' 边缘镜像填充至 14 的整数倍
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # 推理阶段
        if aug_trans is not None:
            aug_imgs = aug_trans(img)
            depth_list = []
            
            for aug_img in aug_imgs:
                input_tensor = patch_transformed(image=aug_img)['image'].unsqueeze(0).float().cuda()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred_patch = outputs.get('regression', None).data.cpu().numpy().squeeze()
                    depth_list.append(pred_patch)
            
            depth_list = aug_trans.recover(depth_list)
            pred_all = np.mean(np.array(depth_list), axis=0)
            std_all = np.std(np.array(depth_list), axis=0)
            
            # 【新增】：从填补后的结果中，精准裁剪回原始输入的尺寸 (1024x1024)
            pred_all = pred_all[:orig_h, :orig_w]
            std_all = std_all[:orig_h, :orig_w]
            
        else:
            input_tensor = patch_transformed(image=img)['image'].unsqueeze(0).float().cuda()
            with torch.no_grad():
                outputs = model(input_tensor)
                pred_all = outputs.get('regression', None).data.cpu().numpy().squeeze()
                
            # 【新增】：从填补后的结果中，精准裁剪回原始输入的尺寸 (1024x1024)
            pred_all = pred_all[:orig_h, :orig_w]

        # 后处理与保存
        image_name = osp.splitext(osp.basename(image_path))[0]
        
        if args.save_tif:
            output_file = osp.join(args.output_dir, 'depth', image_name + '_depth.tif')
            save_tiff_with_geo_info(image_path, output_file, pred_all)
            
            if args.get_conf:
                uncertainty_file = osp.join(args.output_dir, 'uncertainty', image_name + '_uncertainty.tif')
                save_tiff_with_geo_info(image_path, uncertainty_file, std_all)
        else:
            # 兼容非tif情况，输出渲染的JPG
            pred_disp = (pred_all - pred_all.min()) / (pred_all.max() - pred_all.min() + 1e-8)
            if args.grayscale:
                image_out = np.repeat(pred_disp[..., np.newaxis], 3, axis=-1) * 255
                image_out = image_out.astype(np.uint8)
            else:
                depth_colored = (pred_disp * 255).astype(np.uint8)
                image_out = cv2.applyColorMap(depth_colored, cv2.COLORMAP_TURBO) 
                
            cv2.imwrite(osp.join(args.output_dir, 'depth', image_name + '.jpg'), image_out)

if __name__ == '__main__':
    main()
