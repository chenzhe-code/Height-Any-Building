from glob import glob

import h5py
import numpy as np
import os
import re


def read_atl03_beam(filename, beam, verbose=False):
    with h5py.File(os.path.expanduser(filename), "r") as file_id:
        if verbose:
            print(file_id.filename)
            print(list(file_id.keys()))
            print(list(file_id["METADATA"].keys()))

        beams = [key for key in file_id.keys() if bool(re.match(r"gt\d[lr]", key))]
        if beam not in beams:
            print(f"Skipping {beam}: beam group does not exist in {filename}.")
            return None
        if "heights" not in file_id[beam]:
            print(f"Skipping {beam}: heights group does not exist in {filename}.")
            return None
        if "geolocation" not in file_id[beam]:
            print(f"Skipping {beam}: geolocation group does not exist in {filename}.")
            return None

        atl03_mds = {"heights": {}, "geolocation": {}, "bckgrd_atlas": {}}
        for key, value in file_id[beam]["heights"].items():
            atl03_mds["heights"][key] = value[:]
        for key, value in file_id[beam]["geolocation"].items():
            atl03_mds["geolocation"][key] = value[:]
        if "bckgrd_atlas" in file_id[beam]:
            for key, value in file_id[beam]["bckgrd_atlas"].items():
                atl03_mds["bckgrd_atlas"][key] = value[:]

    return atl03_mds


def add_along_track_coordinates(atl03_mds):
    val = atl03_mds
    val["heights"]["x_atc"] = np.zeros_like(val["heights"]["h_ph"], np.float64) + np.nan
    val["heights"]["y_atc"] = np.zeros_like(val["heights"]["h_ph"], np.float64) + np.nan
    val["geolocation"]["ref_elev_all"] = np.zeros_like(val["heights"]["h_ph"], np.float64)

    segment_index_begin = val["geolocation"]["ph_index_beg"] - 1
    segment_pe_count = val["geolocation"]["segment_ph_cnt"]
    segment_distance = val["geolocation"]["segment_dist_x"].astype(np.float64)

    segment_indices, = np.nonzero(
        (segment_index_begin[:-1] >= 0) & (segment_index_begin[1:] >= 0)
    )
    for segment_index in segment_indices:
        idx = segment_index_begin[segment_index]
        first_count = np.copy(segment_pe_count[segment_index])
        second_count = np.copy(segment_pe_count[segment_index + 1])
        count = first_count + second_count

        distance_along_x = np.copy(
            val["heights"]["dist_ph_along"][idx: idx + count].astype(np.float64)
        )
        ref_elev = np.copy(val["geolocation"]["ref_elev"][segment_index])
        distance_along_x[:first_count] += segment_distance[segment_index]
        distance_along_x[first_count:] += segment_distance[segment_index + 1]
        distance_along_y = np.copy(val["heights"]["dist_ph_across"][idx: idx + count])

        val["heights"]["x_atc"][idx: idx + count] = distance_along_x
        val["heights"]["y_atc"][idx: idx + count] = distance_along_y
        val["geolocation"]["ref_elev_all"][idx: idx + first_count] += ref_elev


def filter_by_confidence(signal_conf_ph, photon_data):
    confidence_values = signal_conf_ph[:, 0]
    mask = confidence_values >= 2

    filtered_data = {}
    for key, value in photon_data.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            filtered_data[key] = value[mask]
        else:
            filtered_data[key] = value

    return filtered_data


def empty_photon_data():
    return {
        "x_act": np.array([], dtype=float),
        "y_act": np.array([], dtype=float),
        "h": np.array([], dtype=float),
        "signal_conf_ph": np.empty((0, 0), dtype=int),
        "lat": np.array([], dtype=float),
        "lon": np.array([], dtype=float),
        "ref_elev": np.array([], dtype=float),
    }


def read_plidar(filepath, beam, bounds):
    """Read one PLiDAR/ATL03 beam and return filtered photon arrays."""
    mask_lat = bounds["min_lat"], bounds["max_lat"]
    mask_lon = bounds["min_lon"], bounds["max_lon"]

    atl03_file = glob(filepath)
    if not atl03_file:
        print(f"Skipping {filepath}: file does not exist.")
        return empty_photon_data()

    is2_atl03_mds = read_atl03_beam(atl03_file[0], beam=beam, verbose=False)
    if is2_atl03_mds is None:
        return empty_photon_data()

    required_height_keys = {"lat_ph", "lon_ph", "h_ph", "signal_conf_ph", "dist_ph_along", "dist_ph_across"}
    required_geolocation_keys = {"ph_index_beg", "segment_ph_cnt", "segment_dist_x", "ref_elev"}
    missing_height_keys = required_height_keys - set(is2_atl03_mds["heights"])
    missing_geolocation_keys = required_geolocation_keys - set(is2_atl03_mds["geolocation"])
    if missing_height_keys or missing_geolocation_keys:
        print(
            f"Skipping {beam} in {atl03_file[0]}: missing required datasets "
            f"heights={sorted(missing_height_keys)}, geolocation={sorted(missing_geolocation_keys)}."
        )
        return empty_photon_data()

    add_along_track_coordinates(is2_atl03_mds)

    d3 = is2_atl03_mds
    subset = (
        (d3["heights"]["lat_ph"] >= min(mask_lat))
        & (d3["heights"]["lat_ph"] <= max(mask_lat))
    )
    if mask_lon is not None:
        if mask_lon[0] is not None and mask_lon[1] is None:
            subset = subset & (d3["heights"]["lon_ph"] >= mask_lon[0])
        elif mask_lon[0] is None and mask_lon[1] is not None:
            subset = subset & (d3["heights"]["lon_ph"] <= mask_lon[1])
        else:
            subset = (
                subset
                & (d3["heights"]["lon_ph"] >= min(mask_lon))
                & (d3["heights"]["lon_ph"] <= max(mask_lon))
            )

    photon_data = {
        "x_act": d3["heights"]["x_atc"][subset],
        "y_act": d3["heights"]["y_atc"][subset],
        "h": d3["heights"]["h_ph"][subset],
        "signal_conf_ph": d3["heights"]["signal_conf_ph"][subset],
        "lat": d3["heights"]["lat_ph"][subset],
        "lon": d3["heights"]["lon_ph"][subset],
        "ref_elev": d3["geolocation"]["ref_elev_all"][subset],
    }

    filtered_data = filter_by_confidence(photon_data["signal_conf_ph"], photon_data)
    valid_indices = ~np.isnan(filtered_data["x_act"]) & ~np.isnan(filtered_data["y_act"])
    for key in filtered_data.keys():
        filtered_data[key] = filtered_data[key][valid_indices]

    return filtered_data
