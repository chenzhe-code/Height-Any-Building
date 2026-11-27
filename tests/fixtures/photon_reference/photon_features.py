from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KDTree
from tqdm import tqdm


def cluster_photon_features(photon_data, track_bin_size=1.0):
    coords = photon_data["x_act"]
    if len(coords) < 2:
        print(f"photon_processing skipped: only {len(coords)} candidate photon(s)")
        return {key: np.asarray(value)[:0] for key, value in photon_data.items()}
    sorted_coords = np.sort(coords)
    deltas = np.diff(sorted_coords)
    median_delta = np.median(deltas)
    determine_num = np.mean(deltas)

    print("photon_processing determine_num", determine_num)
    print(median_delta, max(deltas), min(deltas), np.mean(deltas))
    window_radius = int(determine_num * 10)
    if window_radius == 0:
        window_radius = 2
    print("photon_processing window_radius", window_radius)

    coordinates = np.column_stack((photon_data["x_act"], photon_data["h"]))
    kdtree = KDTree(coordinates, leaf_size=40)
    binned_points = bin_and_normalize(photon_data, track_bin_size, determine_num)
    neighborhood_sets = extract_features(binned_points, kdtree, 5)
    clustered_data = cluster_features(neighborhood_sets, photon_data)
    return clustered_data


def bin_and_normalize(data, bin_size, determine_num):
    x_act = data["x_act"]
    bins = np.arange(x_act.min(), x_act.max() + bin_size, bin_size)
    bin_indices = np.digitize(x_act, bins) - 1

    binned = defaultdict(list)
    for idx, bin_idx in enumerate(bin_indices):
        binned[bin_idx].append(idx)

    normalized = {}
    for bin_idx, points in binned.items():
        elevations = data["h"][points]
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(elevations.reshape(-1, 1))

        for cluster_id in np.unique(clustering.labels_):
            if cluster_id == -1:
                continue
            cluster_points = np.array(points)[clustering.labels_ == cluster_id]
            if determine_num < 0.01:
                if len(cluster_points) % 2 == 0:
                    center_idx_1 = cluster_points[len(cluster_points) // 2]
                    center_idx_2 = cluster_points[len(cluster_points) // 2 - 1]
                    normalized.setdefault(bin_idx, []).append(center_idx_1)
                    normalized.setdefault(bin_idx, []).append(center_idx_2)
                else:
                    center_idx = cluster_points[len(cluster_points) // 2]
                    normalized.setdefault(bin_idx, []).append(center_idx)
            else:
                for idx in cluster_points:
                    normalized.setdefault(bin_idx, []).append(idx)
    return normalized


def extract_features(binned_points, kdtree, window_radius=1):
    grid_sets = defaultdict(dict)
    global_sets = {}
    set_counter = 0
    all_grids = sorted(binned_points.keys())

    for current_grid in tqdm(all_grids, desc="Processing grids"):
        start_grid = max(min(all_grids), current_grid - window_radius)
        end_grid = min(max(all_grids), current_grid + window_radius)

        candidate_indices = []
        for grid in range(start_grid, end_grid + 1):
            if grid != current_grid and grid in binned_points:
                candidate_indices.extend(binned_points[grid])

        current_points = binned_points.get(current_grid, [])
        candidate_indices += current_points

        if len(candidate_indices) < 2:
            continue

        candidate_points = np.take(kdtree.data, candidate_indices, axis=0)
        elev_range = np.ptp(candidate_points[:, 1])
        residual_threshold = max(0.5, elev_range * 0.1)
        min_samples = max(2, int(len(candidate_indices) * 0.4))

        try:
            model = RANSACRegressor(
                residual_threshold=residual_threshold,
                min_samples=min_samples,
                max_trials=1000,
                random_state=42,
            ).fit(candidate_points[:, 0].reshape(-1, 1), candidate_points[:, 1])
            slope = model.estimator_.coef_[0]
            if abs(slope) > 0.2:
                continue

            inlier_mask = model.inlier_mask_
            current_set = set(np.array(candidate_indices)[inlier_mask])
        except Exception:
            continue

        current_grid_points = set(current_points)
        grid_set = current_set & current_grid_points
        if not grid_set:
            continue

        set_id = f"set_{set_counter}"
        grid_sets[current_grid][set_id] = grid_set
        global_sets[set_id] = current_set
        set_counter += 1

        for point in grid_set:
            existing_sets = [
                set_id
                for set_id in grid_sets[current_grid]
                if point in grid_sets[current_grid][set_id]
            ]

            if len(existing_sets) > 1:
                largest_set = max(existing_sets, key=lambda value: len(global_sets[value]))
                for set_id in existing_sets:
                    if set_id != largest_set:
                        grid_sets[current_grid][set_id].discard(point)
                        global_sets[set_id].discard(point)

    neighborhood_sets = defaultdict(set)
    for set_id, point_set in global_sets.items():
        for point in point_set:
            neighborhood_sets[point].add(set_id)

    final_sets = {}
    for point, set_ids in neighborhood_sets.items():
        if set_ids:
            largest_set = max(set_ids, key=lambda value: len(global_sets[value]))
            final_sets[point] = global_sets[largest_set]

    return final_sets


def cluster_features(neighborhood_sets, photon_data):
    class UnionFind:
        def __init__(self):
            self.parent = {}

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            self.parent.setdefault(x, x)
            self.parent.setdefault(y, y)
            fx = self.find(x)
            fy = self.find(y)
            if fx != fy:
                self.parent[fy] = fx

    uf = UnionFind()

    for p, neighbors in neighborhood_sets.items():
        for neighbor in neighbors:
            uf.union(p, neighbor)

    clusters = defaultdict(set)
    for p in neighborhood_sets:
        root = uf.find(p)
        clusters[root].add(p)

    valid_indices = set()
    for c in clusters.values():
        valid_indices.update(c)

    mask = np.zeros_like(photon_data["x_act"], dtype=bool)
    mask[list(valid_indices)] = True

    return {k: v[mask] for k, v in photon_data.items()}
