"""Independent beam tasks for optional ordered ATL03 multiprocessing."""
import os
import random
import numpy as np
from threadpoolctl import threadpool_limits

import argparse
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

def _has_photons(data):
    return isinstance(data, dict) and "x_act" in data and len(data["x_act"]) > 0


def _count(data):
    if isinstance(data, dict) and "x_act" in data:
        return len(data["x_act"])
    return 0


_THREAD_LIMIT = None

def initialize_worker(threads):
    global _THREAD_LIMIT
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    _THREAD_LIMIT = threadpool_limits(limits=threads)
    random.seed(1234)
    np.random.seed(1234)


def extract_beam_task(task):
    filepath, beam, bounds, seg_len, debug_counts = task
    all_ground_data = {"h": [], "lat": [], "lon": []}
    all_building_data = {"h": [], "lat": [], "lon": []}
    beam_data = [(beam, read_plidar(filepath, beam, bounds))]
    for beam, filtered_data_by_confidence in beam_data:
        if not _has_photons(filtered_data_by_confidence):
            if debug_counts:
                print(f"[counts] {filepath} {beam}: filtered=0, skipped")
            continue

        if debug_counts:
            print(f"[counts] {filepath} {beam}: filtered={_count(filtered_data_by_confidence)}")

        track_length = max(filtered_data_by_confidence["x_act"]) - min(filtered_data_by_confidence["x_act"])
        segment_count = int(track_length / seg_len) + 1

        for segment_index in range(segment_count):
            start_point = segment_index * seg_len
            end_point = start_point + seg_len
            segment_data = select_segment_by_distance(start_point, end_point, filtered_data_by_confidence)
            if len(segment_data) == 0:
                continue
            if len(segment_data["x_act"]) < 2:
                print(
                    f"Skipping segment {start_point}-{end_point}: "
                    f"only {len(segment_data['x_act'])} photon(s)."
                )
                continue

            ground_building_candidates = reduce_noise_by_elevation_frequency(segment_data, start_point, end_point)
            ransac_data = cluster_photon_features(ground_building_candidates)
            if not ransac_data or all(len(value) == 0 for value in ransac_data.values()):
                if debug_counts:
                    print(
                        f"[counts] {beam} segment {start_point}-{end_point}: "
                        f"dist={_count(segment_data)}, freq1={_count(ground_building_candidates)}, "
                        "ransac=0, skipped"
                    )
                continue

            final_ground_building_data = refine_ground_and_building_candidates(
                segment_data,
                ransac_data,
                1.0,
                0.008,
                10,
            )
            if not _has_photons(final_ground_building_data):
                if debug_counts:
                    print(
                        f"[counts] {beam} segment {start_point}-{end_point}: "
                        f"dist={_count(segment_data)}, freq1={_count(ground_building_candidates)}, "
                        f"ransac={_count(ransac_data)}, refine=0, skipped"
                    )
                continue

            final_ground_building_data = reduce_noise_by_elevation_frequency(
                final_ground_building_data,
                start_point,
                end_point,
            )
            final_ground_data, building_data = cloth_simulation_filter(final_ground_building_data, 5)
            final_ground_data = grow_region_from_seed(
                segment_data,
                final_ground_data,
                1,
                0.02,
                50,
            )
            final_ground_data = reduce_noise_by_elevation_frequency(final_ground_data, start_point, end_point)

            building_data = grow_region_from_seed(
                segment_data,
                building_data,
                0.3,
                0.008,
                1,
            )

            final_building_data = extract_building_photons(building_data, final_ground_data, 3)

            if debug_counts:
                print(
                    f"[counts] {beam} segment {start_point}-{end_point}: "
                    f"dist={_count(segment_data)}, freq1={_count(ground_building_candidates)}, "
                    f"ransac={_count(ransac_data)}, grown={_count(final_ground_building_data)}, "
                    f"ground={_count(final_ground_data)}, building_seed={_count(building_data)}, "
                    f"building={_count(final_building_data)}"
                )

            for key in all_ground_data:
                all_ground_data[key].extend(final_ground_data[key])
            for key in all_building_data:
                all_building_data[key].extend(final_building_data[key])

    return all_ground_data, all_building_data
