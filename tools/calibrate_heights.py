"""Calibrate prediction windows with sparse controls and fuse a height raster."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.merge import merge

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io_utils import partition_blocks
from utils.raster_io import save_geotiff
from utils.rh2ah_converter import RH2AHConverter

# ==============================================================================
# 主执行模块
# ==============================================================================

def main(input_dir, output_dir, window_size=1024, overlap=0.5, CH_img_path=None,
         block_size=None, metric_input=False, fusion_mode="mean",
         skip_ground_filter=False, graph_control_lambda=1.0,
         graph_overlap_lambda=1.0, overlap_mask="all", export_calibrated=False):
    use_conf = fusion_mode == "uncertainty"
    if all(int(x) == -1 for x in block_size.split(',')):
        converter = RH2AHConverter(
            input_dir, output_dir, CH_img_path, window_size, overlap,
            use_conf=use_conf, metric_input=metric_input,
            apply_ground_filter=not skip_ground_filter,
            graph_control_lambda=graph_control_lambda,
            graph_overlap_lambda=graph_overlap_lambda,
            overlap_mask=overlap_mask,
            export_calibrated=export_calibrated,
        )
        converter.run_metric() if metric_input else converter.run()
    else:
        if export_calibrated:
            raise ValueError("Calibrated window export requires whole-tile RH2AH")
        os.makedirs(output_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rh2ah-blocks-", dir=output_dir) as partition_dir:
            partition_blocks(
                input_dir, partition_dir, window_size=window_size, overlap=overlap,
                block_size=block_size, use_conf=use_conf,
            )
            block_names = sorted(os.listdir(partition_dir), key=lambda name: int(name.rsplit("_", 1)[-1]))
            if not block_names:
                raise ValueError("The requested block size produced no complete blocks.")
            absolute_height_files = []
            for name in block_names:
                directory = os.path.join(partition_dir, name)
                print(f"\n===== 正在处理分块: {name} =====")
                converter = RH2AHConverter(
                    os.path.join(directory, "depth"), directory, CH_img_path, window_size, overlap,
                    use_conf=use_conf, metric_input=metric_input,
                    apply_ground_filter=not skip_ground_filter,
                    graph_control_lambda=graph_control_lambda,
                    graph_overlap_lambda=graph_overlap_lambda,
                    overlap_mask=overlap_mask,
                )
                converter.run_metric() if metric_input else converter.run()
                absolute_height_files.append(os.path.join(directory, "AbsoluteHeight", "output.tif"))

            print("\n===== 正在合并所有分块... =====")
            merged_output_path = os.path.join(output_dir, "AbsoluteHeight", "output.tif")
            os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
            with ExitStack() as stack:
                datasets = [stack.enter_context(rasterio.open(file)) for file in absolute_height_files]
                mosaic, out_transform = merge(datasets)
                out_meta = datasets[0].meta.copy()
                out_meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                                transform=out_transform)
                save_geotiff(merged_output_path, mosaic, out_meta)
            print(f"合并后的绝对高度图已保存至: {merged_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work_dir', type=str, required=True, help="用于输入和输出的工作目录。")
    parser.add_argument('--ch_img_path', type=str, help="控制高度GeoTIFF文件的路径。")
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1234, help="Random seed for reproducible RANSAC calibration.")
    parser.add_argument('--down_ratio', type=float, default=1)
    parser.add_argument('--block_size', type=str, default='-1,-1', help="用于分块的尺寸 'rows,cols,overlap'。使用 '-1,-1,-1' 禁用分块。")
    parser.add_argument('--metric_input', action='store_true', help="输入已经是带尺度的高度；跳过稀疏控制标定和尺度拟合，以原始模型尺度直接融合。")
    parser.add_argument('--fusion_mode', choices=('mean', 'uncertainty'), default='mean', help="重叠区域的融合方式。")
    parser.add_argument('--graph-control-lambda', type=float, default=1.0,
                        help="控制约束误差平方和的系数；默认1。")
    parser.add_argument('--graph-overlap-lambda', type=float, default=1.0,
                        help="重叠约束误差平方和的系数；默认1。")
    parser.add_argument('--overlap-mask', choices=('all', 'nonground'), default='all',
                        help="两阶段重叠统计的像元筛选；默认all。")
    parser.add_argument('--export-calibrated', action='store_true',
                        help="Archive floating-point calibrated windows for cross-tile optimization.")
    parser.add_argument(
        '--skip-ground-filter', '--skip_ground_filter',
        dest='skip_ground_filter', action='store_true',
        help="跳过局部最低值地面滤波，直接使用模型输出高度。",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    graph_lambdas = np.asarray([args.graph_control_lambda, args.graph_overlap_lambda])
    if not np.all(np.isfinite(graph_lambdas)) or np.any(graph_lambdas < 0) or not np.any(graph_lambdas > 0):
        parser.error('Graph coefficients must be finite, nonnegative, and not both zero')
    if not args.metric_input and not args.ch_img_path:
        parser.error('--ch_img_path is required unless --metric_input is set')
    
    input_dir = os.path.join(args.work_dir, 'depth')
    output_dir = args.work_dir
    window_size = int(args.window_size / args.down_ratio)
    
    main(input_dir, output_dir, 
         window_size=window_size, 
         overlap=args.overlap, 
         CH_img_path=args.ch_img_path, 
         block_size=args.block_size,
         metric_input=args.metric_input,
         fusion_mode=args.fusion_mode,
         skip_ground_filter=args.skip_ground_filter,
         graph_control_lambda=args.graph_control_lambda,
         graph_overlap_lambda=args.graph_overlap_lambda,
         overlap_mask=args.overlap_mask,
         export_calibrated=args.export_calibrated)
