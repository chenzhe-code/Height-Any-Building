from collections import deque

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree


def _initial_extraction(data, window_size=5.0):
    x = data["x_act"]
    z = data["h"]
    sorted_idx = np.argsort(x, kind="mergesort")
    x_sorted = x[sorted_idx]
    n_points = len(data["x_act"])
    windows = np.arange(x_sorted[0], x_sorted[-1] + window_size, window_size)
    mask = np.zeros(n_points, dtype=bool)

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        window_points = sorted_idx[(x_sorted >= start) & (x_sorted < end)]
        if len(window_points) > 0:
            min_idx = window_points[np.argmin(z[window_points])]
            mask[min_idx] = True

    data_ = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            data_[key] = value[mask]
    return data_


def _elevation_clustering(data, elev_threshold=1.0):
    ground_z = data["h"]
    clustering = DBSCAN(eps=elev_threshold, min_samples=5).fit(ground_z.reshape(-1, 1))
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    cluster_avg_elevations = {}
    for label in unique_labels:
        if label != -1:
            cluster_points = ground_z[labels == label]
            cluster_avg_elevations[label] = np.mean(cluster_points)

    if len(cluster_avg_elevations) > 0:
        main_cluster = min(cluster_avg_elevations, key=cluster_avg_elevations.get)
    else:
        return np.zeros_like(data["h"])

    refined_mask = labels == main_cluster
    data_ = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(refined_mask):
            data_[key] = value[refined_mask]
    return data_


def refine_ground_and_building_candidates(
    dist_chose_data,
    ground_and_building_data,
    max_height_diff=1.0,
    max_slope=0.008,
    search_radius=30,
):
    ground_data_ = ground_and_building_data
    ground_z = ground_data_["h"]
    clustering = DBSCAN(eps=2, min_samples=4).fit(ground_z.reshape(-1, 1))
    labels = clustering.labels_
    unique_labels, counts = np.unique(clustering.labels_, return_counts=True)

    valid_labels = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]

    if len(valid_labels) == 0:
        return np.zeros_like(ground_z)

    main_cluster = valid_labels[np.argmax(valid_counts)]
    refined_mask = labels == main_cluster
    ground_data = {}
    for key, value in ground_data_.items():
        if isinstance(value, np.ndarray) and len(value) == len(refined_mask):
            ground_data[key] = value[refined_mask]

    x_act = dist_chose_data["x_act"]
    h = dist_chose_data["h"]
    data_x_act = ground_data["x_act"]
    data_h = ground_data["h"]

    n_points = len(dist_chose_data["x_act"])
    initial_ground_mask = np.zeros(n_points, dtype=bool)

    for i, x in enumerate(x_act):
        nearest_idx = np.argmin(np.abs(x - data_x_act))
        if np.abs(x - data_x_act[nearest_idx]) == 0:
            initial_ground_mask[i] = True

    points = np.column_stack((x_act, h))
    tree = KDTree(points)
    ground_mask = np.copy(initial_ground_mask)
    visited = np.zeros(n_points, dtype=bool)
    seed_indices = np.where(initial_ground_mask)[0]
    seed_queue = deque(seed_indices.tolist())
    visited[seed_indices] = True

    while seed_queue:
        current_idx = seed_queue.popleft()
        current_point = points[current_idx]
        current_z = h[current_idx]
        neighbors = tree.query_radius([current_point], search_radius)[0]

        for idx in neighbors:
            if not visited[idx]:
                delta_z = abs(h[idx] - current_z)
                if delta_z > max_height_diff:
                    continue

                dx = points[idx][0] - current_point[0]
                horizontal_dist = abs(dx)
                if horizontal_dist < 1e-6:
                    continue

                slope = delta_z / horizontal_dist
                if slope > max_slope:
                    continue

                ground_mask[idx] = True
                visited[idx] = True
                seed_queue.append(idx)

    final_ground_building_data = {}
    for key, value in dist_chose_data.items():
        if isinstance(value, np.ndarray) and len(value) == len(ground_mask):
            final_ground_building_data[key] = value[ground_mask]

    final_ground_building_ungrow = {}
    source_points = np.column_stack((ground_and_building_data["x_act"], ground_and_building_data["h"]))
    seed_points = np.column_stack((data_x_act, data_h))
    tree = cKDTree(seed_points)
    distances, _ = tree.query(source_points, k=1)
    mask = distances < 1e-6
    for key, value in ground_and_building_data.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            final_ground_building_ungrow[key] = value[~mask]

    final_buildings = grow_region_from_seed(
        dist_chose_data,
        final_ground_building_ungrow,
        0.5,
        0.005,
        5,
    )
    for key, value in final_buildings.items():
        final_ground_building_data[key] = np.concatenate([final_ground_building_data[key], value])

    return final_ground_building_data


def grow_region_from_seed(
    dist_chose_data,
    ground_and_building_data,
    max_height_diff=1.0,
    max_slope=0.1,
    search_radius=50,
):
    x_act = dist_chose_data["x_act"]
    h = dist_chose_data["h"]
    data_x_act = ground_and_building_data["x_act"]

    n_points = len(dist_chose_data["x_act"])
    initial_ground_mask = np.zeros(n_points, dtype=bool)

    for i, x in enumerate(x_act):
        if data_x_act.size == 0:
            continue
        nearest_idx = np.argmin(np.abs(x - data_x_act))
        if np.abs(x - data_x_act[nearest_idx]) == 0:
            initial_ground_mask[i] = True

    points = np.column_stack((x_act, h))
    tree = KDTree(points)
    ground_mask = np.copy(initial_ground_mask)
    visited = np.zeros(n_points, dtype=bool)
    seed_indices = np.where(initial_ground_mask)[0]
    seed_queue = deque(seed_indices.tolist())
    visited[seed_indices] = True

    while seed_queue:
        current_idx = seed_queue.popleft()
        current_point = points[current_idx]
        current_z = h[current_idx]
        neighbors = tree.query_radius([current_point], search_radius)[0]

        for idx in neighbors:
            if not visited[idx]:
                delta_z = abs(h[idx] - current_z)
                if delta_z > max_height_diff:
                    continue

                dx = points[idx][0] - current_point[0]
                horizontal_dist = abs(dx)
                if horizontal_dist < 1e-6:
                    continue

                slope = delta_z / horizontal_dist
                if slope > max_slope:
                    continue

                ground_mask[idx] = True
                visited[idx] = True
                seed_queue.append(idx)

    final_ground_building_data = {}
    for key, value in dist_chose_data.items():
        if isinstance(value, np.ndarray) and len(value) == len(ground_mask):
            final_ground_building_data[key] = value[ground_mask]
    return final_ground_building_data


def extract_building_photons(mixed_data, ground_data, height_threshold=3.0, grid_size=1):
    ground_x = ground_data["x_act"]
    ground_h = ground_data["h"]
    x_min, x_max = np.min(ground_x), np.max(ground_x)
    grid_x = np.arange(x_min, x_max, grid_size)

    f = interp1d(ground_x, ground_h, kind="linear", fill_value="extrapolate")
    ground_grid = f(grid_x)
    grid_tree = cKDTree(grid_x.reshape(-1, 1))

    points = mixed_data["x_act"].reshape(-1, 1)
    mixed_h = mixed_data["h"]
    k = 5
    distances, indices = grid_tree.query(points, k)

    valid_rows = ~np.isinf(distances).any(axis=1)
    distances = distances[valid_rows]
    indices = indices[valid_rows]

    neighbor_heights = ground_grid[indices]
    ground_h_interp_avg = np.mean(neighbor_heights, axis=1) if len(neighbor_heights) > 0 else np.zeros_like(mixed_h)

    height_diff = mixed_h - ground_h_interp_avg
    building_mask = (height_diff >= height_threshold) & (~np.isnan(height_diff))

    return {k: v[building_mask] for k, v in mixed_data.items()}
