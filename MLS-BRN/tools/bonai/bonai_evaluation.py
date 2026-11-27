#!/usr/bin/env python3
"""Export MLS-BRN prediction CSV files; no training or benchmark evaluation.

Preserves the production city=test, --skip-metrics parser and dump settings.
The BSPklParser and bs_csv_dump implementations come from the external bstool
backend with the maintained MLS-BRN/bstool package.
"""
import argparse
import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / 'bstool'), str(BACKEND)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pkl_file_path')
    p.add_argument('csv_save_path')
    p.add_argument('--anno_file', required=True)
    p.add_argument('--test_image_dir')
    p.add_argument('--city', choices=['test'], default='test')
    p.add_argument('--skip-metrics', action='store_true')
    args = p.parse_args()
    import bstool
    import mmcv
    results = mmcv.load(args.pkl_file_path)
    if not results:
        p.error('Prediction PKL is empty; expected one result record per image')
    os.makedirs(args.csv_save_path, exist_ok=True)
    # Empty leading patches have None attributes; use the first detected patch
    # to select the exporter, while retaining the production with_height=False
    # parser path (that path still exports the predicted Height field).
    example = next((item for item in results
                    if item is not None and item[0] is not None
                    and any(len(boxes) for boxes in item[0])), results[0])
    with_offset = example is not None and len(example) > 2 and example[2] is not None
    parser_class = bstool.BSPklParser if with_offset else bstool.BSPklParser_Without_Offset
    parser = parser_class(
        args.anno_file, args.pkl_file_path, iou_threshold=0.1,
        score_threshold=0.4, min_area=500, with_offset=with_offset,
        with_height=False, gt_roof_csv_file=None,
        replace_pred_roof=False, replace_pred_offset=False,
        offset_model='footprint2roof', merge_splitted=False,
    )
    bstool.bs_csv_dump(parser.objects,
        os.path.join(args.csv_save_path, 'roof_pred.csv'),
        os.path.join(args.csv_save_path, 'footprint_offset.csv'),
        os.path.join(args.csv_save_path, 'footprint_direct.csv'))


if __name__ == '__main__':
    main()
