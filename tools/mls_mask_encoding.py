"""Exact binary-mask RLE encoding using the nonempty rectangle.

Retains the original full-image COCO RLE, including compressed counts bytes.
Only the CPU representation conversion changes; model predictions do not.
"""
import cv2
import numpy as np
import pycocotools.mask as mask_utils


def encode_binary_mask(mask):
    """Encode a 2-D bool mask with identical full-image COCO RLE counts."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        return mask_utils.encode(np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
    height, width = mask.shape
    x, y, crop_width, crop_height = cv2.boundingRect(mask.view(np.uint8))
    if not crop_width:
        counts = [height * width]
    else:
        crop = mask[y:y+crop_height, x:x+crop_width]
        # Signed differences identify every column's foreground intervals.
        padded = np.zeros((crop_height + 2, crop_width), dtype=np.int8)
        padded[1:-1] = crop
        changes = np.diff(padded, axis=0).T
        start_cols, start_rows = np.nonzero(changes == 1)
        end_cols, end_rows = np.nonzero(changes == -1)
        starts = (start_cols.astype(np.int64) + x) * height + y + start_rows
        ends = (end_cols.astype(np.int64) + x) * height + y + end_rows
        # Adjacent full-height columns can form one foreground run.
        adjacent = starts[1:] == ends[:-1]
        starts = starts[np.r_[True, ~adjacent]]
        ends = ends[np.r_[~adjacent, True]]
        runs = np.empty(2 * len(starts), dtype=np.int64)
        runs[0::2] = starts - np.r_[0, ends[:-1]]
        runs[1::2] = ends - starts
        counts = runs.tolist()
        trailing = height * width - int(ends[-1])
        if trailing:
            counts.append(trailing)
    return mask_utils.frPyObjects({'size': [height, width], 'counts': counts}, height, width)


def encode_mask_results(mask_results):
    """Drop-in counterpart of mmdet.core.mask.utils.encode_mask_results."""
    if isinstance(mask_results, tuple):
        masks, scores = mask_results
    else:
        masks = mask_results
    encoded = [[encode_binary_mask(mask) for mask in group] for group in masks]
    return (encoded, scores) if isinstance(mask_results, tuple) else encoded
