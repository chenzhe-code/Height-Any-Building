import argparse
import concurrent.futures
import multiprocessing
import warnings

import h5py

from plidar_toolbox.csf_filter import cloth_simulation_filter
from plidar_toolbox.denoise import reduce_noise_by_elevation_frequency, select_segment_by_distance
from plidar_toolbox.ground_building import (
    extract_building_photons,
    grow_region_from_seed,
    refine_ground_and_building_candidates,
)
from plidar_toolbox.io_utils import check_h5_files, get_geotiff_bounds, get_shapefile_bounds, save_csv
from plidar_toolbox.photon_features import cluster_photon_features
from plidar_toolbox.readers import read_plidar

warnings.filterwarnings("ignore", category=UserWarning)

from plidar_toolbox.beam_processing import extract_beam_task, initialize_worker

DEFAULT_BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")


def _resolve_bounds(bound_file):
    if bound_file.endswith(".shp"):
        return get_shapefile_bounds(bound_file)
    if bound_file.endswith(".tif"):
        return get_geotiff_bounds(bound_file)
    raise ValueError("Only .shp and .tif boundary files are supported.")


def _has_photons(data):
    return isinstance(data, dict) and "x_act" in data and len(data["x_act"]) > 0


def _count(data):
    if isinstance(data, dict) and "x_act" in data:
        return len(data["x_act"])
    return 0


def _available_beams(filepath, beams):
    with h5py.File(filepath, "r") as h5_file:
        return [
            beam
            for beam in beams
            if h5_file.get(f"{beam}/heights/lat_ph") is not None
            and h5_file.get(f"{beam}/heights/lon_ph") is not None
        ]


def extract_heights(
    bound_file,
    plidar_folder,
    save_path,
    seg_len=1000,
    debug_counts=False,
    beams=None,
    workers=1,
    worker_threads=1,
):
    """Run the PLiDAR height extraction workflow and save ground/building photons."""
    bounds = _resolve_bounds(bound_file)
    beams = tuple(beams or DEFAULT_BEAMS)
    all_ground_data = {"h": [], "lat": [], "lon": []}
    all_building_data = {"h": [], "lat": [], "lon": []}

    if workers < 1 or worker_threads < 1:
        raise ValueError("workers and worker_threads must be positive")
    plidar_files = check_h5_files(plidar_folder, bounds, beams=beams)
    tasks = []
    for filepath in plidar_files:
        print("Processing file:", filepath)
        tasks.extend((filepath, beam, bounds, seg_len, debug_counts)
                     for beam in _available_beams(filepath, beams))

    def collect(results):
        for ground, building in results:
            for key in all_ground_data:
                all_ground_data[key].extend(ground[key])
            for key in all_building_data:
                all_building_data[key].extend(building[key])

    if workers == 1 or not tasks:
        collect(map(extract_beam_task, tasks))
    else:
        # Spawn avoids inheriting HDF5/OpenMP state; map preserves file/beam order.
        print(f"Parallel ATL03: {workers} workers, {worker_threads} numerical threads per worker")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(worker_threads,),
        ) as pool:
            collect(pool.map(extract_beam_task, tasks, chunksize=1))

    print(f"Total ground photons: {len(all_ground_data['h'])}")
    print(f"Total building photons: {len(all_building_data['h'])}")
    print(f"Total saved rows: {len(all_ground_data['h']) + len(all_building_data['h'])}")
    save_csv(all_ground_data, all_building_data, save_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract ground and building photons from PLiDAR/ATL03 data.")
    parser.add_argument("--bound-file", "--bound_file", dest="bound_file", type=str, required=True)
    parser.add_argument("--plidar-folder", "--plidar_folder", dest="plidar_folder", type=str, required=True)
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=str, required=True)
    parser.add_argument("--seg-len", "--seg_len", dest="seg_len", type=int, default=1000)
    parser.add_argument("--beams", nargs="+", default=list(DEFAULT_BEAMS))
    parser.add_argument("--debug-counts", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Independent beam processes; default 1 preserves serial execution.")
    parser.add_argument("--worker-threads", type=int, default=1, help="Numerical threads per parallel worker.")
    return parser


def main(bound_file=None, plidar_folder=None, save_path=None, seg_len=1000, debug_counts=False, beams=None, workers=1, worker_threads=1):
    if bound_file is None or plidar_folder is None or save_path is None:
        args = build_arg_parser().parse_args()
        bound_file = args.bound_file
        plidar_folder = args.plidar_folder
        save_path = args.save_path
        seg_len = args.seg_len
        debug_counts = args.debug_counts
        beams = args.beams
        workers = args.workers
        worker_threads = args.worker_threads

    extract_heights(bound_file, plidar_folder, save_path, seg_len, debug_counts, beams, workers, worker_threads)


if __name__ == "__main__":
    main()
