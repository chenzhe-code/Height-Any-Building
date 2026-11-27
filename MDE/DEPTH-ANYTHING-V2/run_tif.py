import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import rasterio
from rasterio.windows import Window

from depth_anything_v2.dpt import DepthAnythingV2

from img_aug.build_aug import AugTransform


def read_geotiff(file_path):
    with rasterio.open(file_path) as src:
        return src.read(), src.meta
    
def save_geotiff(file_path, data, meta):
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_tif', default=False, dest= 'save_tif', action='store_true', help='save as the geotiff format')
    parser.add_argument('--get-conf', dest='get_conf', default=False, action='store_true', help='get the pred conf by std')
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(os.path.join(args.outdir, 'depth'), exist_ok=True)
    if args.get_conf:
        os.makedirs(os.path.join(args.outdir, 'uncertainty'), exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    aug_trans = AugTransform() if args.get_conf else None
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image, meta = read_geotiff(filename)
        raw_image = np.transpose(raw_image, [1, 2, 0])

        if aug_trans is not None:
            aug_imgs = aug_trans(raw_image)
            depth_list = [depth_anything.infer_image(aug_img, args.input_size) for aug_img in aug_imgs]
            depth_list = aug_trans.recover(depth_list)

            depth = np.mean(np.array(depth_list), axis=0)

            uncertainty = np.std(np.array(depth_list), axis=0)
            
            # print('max:', np.max(depth))
            # print('min:', np.min(depth))

            # depth_list.append(depth)
            # depth_list.append(uncertainty)

            # # norm_depth_list.append(depth)
            # # norm_depth_list.append(uncertainty)

            # fig, axs = plt.subplots(2, 5, figsize = (15, 15))
            # for i, recover_depth in enumerate(depth_list):
            #     row, col = divmod(i, 5)
            #     A = axs[row, col].imshow(recover_depth, cmap='viridis', interpolation='nearest')
            #     axs[row, col].set_title(f'img{i}')
            #     fig.colorbar(A, ax=axs[row, col], shrink=0.3)

            # plt.tight_layout()
            # plt.show()

        else:
            depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            if args.save_tif:
                output_file = os.path.join(args.outdir, 'depth', os.path.splitext(os.path.basename(filename))[0] + '_depth.tif')
                save_geotiff(output_file, depth.transpose(2,0,1), meta)
            else:
                cv2.imwrite(os.path.join(args.outdir, 'depth', os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)

        if args.get_conf and args.save_tif:
            uncertainty = np.repeat(uncertainty[..., np.newaxis], 3, axis=-1)
            output_file = os.path.join(args.outdir, 'uncertainty', os.path.splitext(os.path.basename(filename))[0] + '_uncertainty.tif')
            save_geotiff(output_file, uncertainty.transpose(2,0,1), meta)