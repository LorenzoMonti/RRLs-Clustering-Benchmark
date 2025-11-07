import numpy as np
from scipy.spatial import distance

"""
The implementation is based on the paper "SS-DBSCAN: Semi-Supervised Density-Based Spatial 
Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data" 
https://ieeexplore.ieee.org/document/10670579  and its official repository: 
https://github.com/TibaZaki/SS_DBSCAN
"""

def is_important_astrophysics(point_features, max_values, min_values):
    """
    Determines if a point is "important" based on astrophysical heuristics.
    A point is important if its Lz is below a certain threshold or its Energy
    is above a certain threshold.

    Args:
        point_features (np.ndarray): Features of the current point (Lz, Energy, ...).
        max_values (np.ndarray): Maximum values for each feature in the dataset.
        min_values (np.ndarray): Minimum values for each feature in the dataset.

    Returns:
        bool: True if the point is important, False otherwise.
    """
    lz_index, energy_index = 0, 1
    lz, energy = point_features[lz_index], point_features[energy_index]
    lz_threshold = min_values[lz_index] + 0.15 * (max_values[lz_index] - min_values[lz_index])
    energy_threshold = max_values[energy_index] - 0.15 * (max_values[energy_index] - min_values[energy_index])
    return lz <= lz_threshold or energy >= energy_threshold


def _region_query(P, eps, distance_matrix):
    """
    Performs a region query to find all points within a given distance (eps) of point P.

    Args:
        P (int): Index of the reference point.
        eps (float): The maximum distance to consider.
        distance_matrix (np.ndarray): Precomputed distance matrix of the dataset.

    Returns:
        list: A list of indices of neighbor points.
    """
    return [Pn for Pn, dist in enumerate(distance_matrix[P]) if dist < eps]


def _grow_cluster_astro(labels, P, NeighborPts, C, eps, MinPts, Features, dm, max_vals, min_vals):
    """
    Expands a cluster from a core point, incorporating the astrophysical heuristic.

    Args:
        labels (np.ndarray): Array of cluster labels for each point.
        P (int): Index of the core point.
        NeighborPts (list): List of neighbors of point P.
        C (int): Current cluster ID.
        eps (float): The maximum distance to consider.
        MinPts (int): The minimum number of points required to form a dense region.
        Features (np.ndarray): The dataset features.
        dm (np.ndarray): Precomputed distance matrix.
        max_vals (np.ndarray): Max feature values.
        min_vals (np.ndarray): Min feature values.
    """
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] in [-1, 0]: # If point is noise or unvisited
            if labels[Pn] == 0: # If unvisited
                PnNeighborPts = _region_query(Pn, eps, dm)
                # If it's a core point AND satisfies the heuristic, expand from it
                if len(PnNeighborPts) >= MinPts and is_important_astrophysics(Features[Pn], max_vals, min_vals):
                    for neighbor in PnNeighborPts:
                        if neighbor not in NeighborPts:
                            NeighborPts.append(neighbor)
            labels[Pn] = C
        i += 1


def run_heuristic_ssdscan(X, eps, min_samples):
    """
    Executes SSDBSCAN clustering with an astrophysical heuristic.

    Args:
        X (np.ndarray): The input data.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a
                           point to be considered as a core point.

    Returns:
        np.ndarray: Cluster labels for each point, where -1 indicates noise.
    """
    print("\n--- [2/3] Executing SSDBSCAN (Heuristic)... ---")
    n_points = X.shape[0]
    labels = np.zeros(n_points, dtype=int)
    C = 0  # Cluster counter

    # print("Calculating Euclidean distance matrix...")
    dm = distance.cdist(X, X, 'euclidean')
    max_vals, min_vals = np.max(X, axis=0), np.min(X, axis=0)

    for P in range(n_points):
        if labels[P] == 0:  # If point P has not been visited
            NeighborPts = _region_query(P, eps, dm)
            if len(NeighborPts) < min_samples:
                labels[P] = -1  # Mark as noise
            else:
                C += 1  # Next cluster ID
                _grow_cluster_astro(labels, P, NeighborPts, C, eps, min_samples, X, dm, max_vals, min_vals)

    # Re-label clusters to be 0-indexed, with -1 for noise
    unique_labels = np.unique(labels[labels != -1])
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map.get(l, -1) for l in labels])