import rasterio
import os
import argparse
import cv2

if __package__:
    from utils.raster_io import read_geotiff, save_geotiff
    from utils.window_grid import window_positions
else:  # Direct execution: python utils/split_image.py
    from raster_io import read_geotiff, save_geotiff
    from window_grid import window_positions

def split_image(input_file, output_dir, window_size, overlap, down_ratio):
    if down_ratio <= 0:
        raise ValueError(f"down_ratio must be positive, got {down_ratio}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Rebuilding a generated patch directory must not retain positions from an
    # older edge-dropping grid.
    for filename in os.listdir(output_dir):
        if filename.startswith("window_") and filename.lower().endswith((".tif", ".tiff")):
            os.remove(os.path.join(output_dir, filename))

    with rasterio.open(input_file) as dataset:
        width = dataset.width
        height = dataset.height
        crs = dataset.crs

        count = 0

        for x, y in window_positions(width, height, window_size, overlap):
            window = rasterio.windows.Window(x, y, window_size, window_size)
            transform_window = rasterio.windows.transform(window, dataset.transform)
            cropped_image = dataset.read(window=window)
            _, p_height, p_width = cropped_image.shape
            output_height = int(p_height // down_ratio)
            output_width = int(p_width // down_ratio)
            if output_height <= 0 or output_width <= 0:
                raise ValueError(
                    f"down_ratio {down_ratio} produces an empty output patch"
                )
            cropped_image = cv2.resize(
                cropped_image.transpose(1, 2, 0),
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)

            output_file = os.path.join(
                output_dir,
                f'window_{count}_x{int(x // down_ratio)}_y{int(y // down_ratio)}.tif',
            )

            profile = dict(
                driver='GTiff',
                height=output_height,
                width=output_width,
                count=dataset.count,
                dtype=cropped_image.dtype,
                crs=crs,
                transform=transform_window * transform_window.scale(down_ratio, down_ratio),
            )
            save_geotiff(output_file, cropped_image, profile)

            count += 1
            print(count, f"Saved {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_dir', type=str, default='./split_windows')
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--down_ratio', type=float, default=1, help='downsample the original image')

    args = parser.parse_args()

    input_file = args.input_path
    output_dir = args.output_dir
    window_size = args.window_size
    overlap = args.overlap
    down_ratio = args.down_ratio
    split_image(input_file, output_dir, window_size, overlap, down_ratio)
