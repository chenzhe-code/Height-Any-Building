import numpy as np


def select_segment_by_distance(start_point, end_point, filtered_data):
    filtered_data_copy = filtered_data.copy()
    if len(filtered_data_copy["x_act"]) == 0:
        print("The selected area has no photon data.")
        return 0

    min_act = min(filtered_data_copy["x_act"])
    filtered_data_copy["x_act"] = filtered_data_copy["x_act"] - min_act

    dist_value = filtered_data_copy["x_act"]
    mask = (dist_value >= start_point) & (dist_value <= end_point)
    dist_chose_data = {}
    for key, value in filtered_data_copy.items():
        dist_chose_data[key] = value[mask]

    return dist_chose_data


def reduce_noise_by_elevation_frequency(photon_data, start_point, end_point, num_bins=10):
    coords = photon_data["x_act"]
    if len(coords) < 2:
        print(f"frequency filter passthrough: only {len(coords)} photon(s)")
        return {key: np.asarray(value).copy() for key, value in photon_data.items()}
    sorted_coords = np.sort(coords)
    deltas = np.diff(sorted_coords)
    median_delta = np.median(deltas)
    determine_num = np.mean(deltas)

    print("frequency_chose determine_num", determine_num)
    print(median_delta, max(deltas), min(deltas), np.mean(deltas))
    window_size = int(determine_num * 50) * 0 + 2

    if window_size == 0:
        window_size = 2

    print("window_size", window_size)
    valid_mask = (photon_data["x_act"] >= start_point) & (photon_data["x_act"] <= end_point)
    valid_coords = photon_data["x_act"][valid_mask]
    valid_heights = photon_data["h"][valid_mask]
    sorted_idx = np.argsort(valid_coords)
    sorted_values = valid_coords[sorted_idx]
    keep_mask = np.zeros_like(valid_mask, dtype=bool)
    segment_length = end_point - start_point

    for seg_start in np.arange(start_point, end_point, segment_length):
        seg_end = min(seg_start + segment_length, end_point)
        for win_start in np.arange(seg_start, seg_end, window_size):
            win_end = min(win_start + window_size, seg_end)
            # Half-open windows and the original argsort order are unchanged.
            lo, hi = np.searchsorted(sorted_values, [max(seg_start, win_start), min(seg_end, win_end)], side="left")
            window_indices = sorted_idx[lo:hi]

            if len(window_indices) == 0:
                continue

            elevations = valid_heights[window_indices]
            z_range = np.ptp(elevations)

            if z_range < 0.5:
                keep_mask[window_indices] = True
                continue

            bins = np.linspace(elevations.min(), elevations.max(), num_bins + 1)
            bin_indices = np.digitize(elevations, bins) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            counts = np.bincount(bin_indices, minlength=num_bins)
            max_bin = np.argmax(counts)
            bin_mask = (
                (bin_indices >= max(0, max_bin - 1))
                & (bin_indices <= min(num_bins - 1, max_bin + 1))
            )
            keep_mask[window_indices[bin_mask]] = True

    coarse_noise_reduction = {
        key: value[valid_mask][keep_mask[valid_mask]]
        for key, value in photon_data.items()
    }
    return coarse_noise_reduction
