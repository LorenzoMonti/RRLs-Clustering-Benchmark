import pandas as pd
import numpy as np
import os
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from joblib import Parallel, delayed

# Import algorithms for evaluation functions
from algorithms.ssd_heuristic import run_heuristic_ssdscan
from algorithms.cdbscan import CDBSCAN
from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory


# ==============================================================================
# === Data Loading and Preprocessing ===========================================
# ==============================================================================

def read_csv_dataset(data_path):
    """Reads a CSV dataset and adds a unique ID column."""
    df = pd.read_csv(data_path)
    df['id'] = np.arange(len(df))
    mapping_id = df[["source_id", "id"]]
    return df.to_numpy(), mapping_id

def read_csv_centroids(centroids_path):
    """Reads centroid data from a CSV file."""
    return pd.read_csv(centroids_path).to_numpy()

def select_features(dataset, features):
    """Selects specific feature columns from a dataset."""
    return dataset[:, features]

def energy_preprocessing(dataset, times=1000):
    """Performs energy preprocessing by shuffling and subtracting the mean."""
    from sklearn.utils import shuffle
    columns = dataset[:, 1].shape[0]
    shuffle_energy = [shuffle(dataset[:, 1]) for _ in range(times)]
    mean_shuffled_energy = np.mean(shuffle_energy, axis=0)
    dataset[:, 1] -= mean_shuffled_energy
    return dataset

# ==============================================================================
# === Seed and Constraint Generation ===========================================
# ==============================================================================

def create_seeds_dict_with_indices(path_clusters_csv, path_cluster_directory, original_dataset_df, seed_percentage=100, scaler=None):
    """
    Creates a dictionary of seed points and their original indices for each known cluster.
    """
    if scaler is None:
        raise ValueError("A pre-fitted scaler must be provided.")

    df_centroids = pd.read_csv(os.path.join(path_clusters_csv, "rrls_clusters.csv"), header=None, skiprows=1)
    seeds_info_dict = {}

    for _, row in df_centroids.iterrows():
        cluster_id = row[0]
        centroid_coords = np.array([row[3], row[2], row[4]]) # Lz, Energy, FeH
        cluster_file_path = os.path.join(path_cluster_directory, f"rrls_{cluster_id}.csv")

        if os.path.exists(cluster_file_path):
            df_cluster = pd.read_csv(cluster_file_path, header=None, skiprows=1)
            cluster_points_coords = df_cluster[[2, 1, 3]].values
            
            # This logic assumes indices from df_cluster align with original_dataset_df
            cluster_points_indices = df_cluster.index.values

            if seed_percentage > 0 and len(cluster_points_coords) > 0:
                num_seeds = max(1, int(len(cluster_points_coords) * (seed_percentage / 100.0)))
                distances = cdist(cluster_points_coords, centroid_coords.reshape(1, -1))[:, 0]
                point_info = sorted(zip(cluster_points_coords, distances, cluster_points_indices), key=lambda x: x[1])
                indices_to_sample = np.linspace(0, len(point_info) - 1, num_seeds, dtype=int)
                selected_seeds = [point_info[i] for i in indices_to_sample]
                selected_seed_coords = [s[0] for s in selected_seeds]
                selected_seed_indices = [s[2] for s in selected_seeds]
            else:
                selected_seed_coords, selected_seed_indices = [], []
            
            selected_seed_coords_np = np.array(selected_seed_coords)
            seeds_info_dict[cluster_id] = {
                'centroid_coords_scaled': scaler.transform([centroid_coords])[0],
                'seed_coords_scaled': scaler.transform(selected_seed_coords_np) if len(selected_seed_coords_np) > 0 else np.empty((0,3)),
                'seed_indices': np.array(selected_seed_indices, dtype=int)
            }
    return seeds_info_dict, scaler

# ==============================================================================
# === Evaluation and Optimization Helpers ======================================
# ==============================================================================

def _evaluate_ssd_config(params, X, ground_truth_labels, mask_meaningful):
    """Helper for parallel optimization of SSDBSCAN."""
    eps, min_samples = params['eps'], params['min_samples']
    labels = run_heuristic_ssdscan(X, eps=eps, min_samples=min_samples)
    ari = adjusted_rand_score(ground_truth_labels[mask_meaningful], labels[mask_meaningful])
    print(f"  SSDBSCAN Test: eps={eps}, min_samples={min_samples} -> ARI: {ari:.4f}")
    return ari, params

def _evaluate_cdb_config(params, X, must_link, cannot_link, ground_truth_labels, mask_meaningful):
    """Helper for parallel optimization of C-DBSCAN."""
    eps, min_pts = params['eps'], params['min_pts']
    model = CDBSCAN(eps=eps, min_pts=min_pts)
    model.fit(X, must_link=must_link, cannot_link=cannot_link)
    ari = adjusted_rand_score(ground_truth_labels[mask_meaningful], model.labels_[mask_meaningful])
    print(f"  C-DBSCAN Test: eps={eps}, min_pts={min_pts} -> ARI: {ari:.4f}")
    return ari, params

def _evaluate_climb_config(params, X, seeds_for_climb, ground_truth_labels, mask_known_points, known_labels):
    """Helper to optimize CLiMB parameters, evaluating Phase 1 on known points."""
    phase1_params = {k: v for k, v in params.items() if not k.startswith('exploratory_')}
    phase2_params = {'eps': params.get('exploratory__eps', 0.190), 'min_samples': params.get('exploratory__min_samples', 24)}

    climb_instance = CLiMB(
        constrained_clusters=len(seeds_for_climb), seed_points=seeds_for_climb,
        distance_metric="mahalanobis", metric_params={'VI': np.linalg.inv(np.cov(X.T))},
        **phase1_params, exploratory_algorithm=DBSCANExploratory(**phase2_params)
    )
    climb_instance.fit(X, known_labels, is_slight_movement=False)
    
    pred_labels_phase1 = climb_instance.constrained_labels
    true_labels_known = ground_truth_labels[mask_known_points]
    pred_labels_known = pred_labels_phase1[mask_known_points]
    ari = adjusted_rand_score(true_labels_known, pred_labels_known)
    print(f"  CLiMB Test: {params} -> ARI (Known Part): {ari:.4f}")
    return ari, params

def _evaluate_climb_exploratory_config(params, unassigned_points):
    """Helper to optimize CLiMB's exploratory phase parameters using Davies-Bouldin Index."""
    eps, min_samples = params['eps'], params['min_samples']
    exploratory_algo = DBSCANExploratory(eps=eps, min_samples=min_samples)
    labels = exploratory_algo.fit_predict(unassigned_points)
    mask_clustered = (labels != -1)
    n_clusters = len(np.unique(labels[mask_clustered]))

    if n_clusters > 1 and np.sum(mask_clustered) > 1:
        dbi_score = davies_bouldin_score(unassigned_points[mask_clustered], labels[mask_clustered])
        print(f"  Phase 2 Test: eps={eps}, min_samples={min_samples} -> davies_bouldin: {dbi_score:.4f}")
        return dbi_score, params
    return float('inf'), params

def optimize_benchmarks(X, ground_truth_labels, centroids_dataset, must_link, cannot_link, seeds_for_climb, known_labels):
    """Performs hyperparameter search for all comparison algorithms."""
    print("\n--- [OPTIMIZATION PHASE] Searching for Optimal Hyperparameters ---")
    known_cluster_ids = np.unique(centroids_dataset[:, 0]).astype(int)
    mask_known_points = np.isin(ground_truth_labels, known_cluster_ids)
    mask_all_meaningful_gt = (ground_truth_labels != 0)

    # CLiMB Optimization
    print("\n--- Optimizing CLiMB... ---")
    climb_p1_grid = [{'density_threshold': dt, 'distance_threshold': dist_t, 'radial_threshold': rad_t, 'convergence_tolerance': conv_t} 
                     for dt in [0.005, 0.01] for dist_t in [0.3, 0.5] for rad_t in [0.1, 0.4] for conv_t in [0.01, 0.03]]
    climb_full_grid = [{**p1, 'exploratory__eps': 0.190, 'exploratory__min_samples': 24} for p1 in climb_p1_grid]
    
    results_p1 = Parallel(n_jobs=-1)(delayed(_evaluate_climb_config)(p, X, seeds_for_climb, ground_truth_labels, mask_known_points, known_labels) for p in climb_full_grid)
    best_ari_p1, best_params_p1_full = max(results_p1, key=lambda item: item[0])
    best_params_p1 = {k: v for k, v in best_params_p1_full.items() if not k.startswith('exploratory__')}
    print(f"Best Phase 1 params: {best_params_p1} (ARI: {best_ari_p1:.4f})")
    
    climb_for_p2 = CLiMB(
        constrained_clusters=len(seeds_for_climb), seed_points=seeds_for_climb, **best_params_p1,
        distance_metric="mahalanobis", metric_params={'VI': np.linalg.inv(np.cov(X.T))},
        exploratory_algorithm=DBSCANExploratory(eps=0.1, min_samples=5)
    )
    climb_for_p2.fit(X, known_labels, is_slight_movement=False)
    unassigned_points = climb_for_p2.unassigned_points

    if unassigned_points is not None and len(unassigned_points) > 0:
        climb_p2_grid = [{'eps': eps, 'min_samples': ms} for eps in [0.15, 0.19, 0.24] for ms in [16, 24, 50]]
        results_p2 = Parallel(n_jobs=-1)(delayed(_evaluate_climb_exploratory_config)(p, unassigned_points) for p in climb_p2_grid)
        best_dbi_p2, best_params_p2 = min(results_p2, key=lambda item: item[0])
        print(f"Best Phase 2 params: {best_params_p2} (DBI: {best_dbi_p2:.4f})")
    else:
        best_params_p2 = {'eps': 0.190, 'min_samples': 24}
    
    # SSDBSCAN Optimization
    print("\n--- Optimizing SSDBSCAN (Heuristic)... ---")
    ssd_grid = [{'eps': eps, 'min_samples': ms} for eps in [0.15, 0.19, 0.24] for ms in [16, 24, 50]]
    results_ssd = Parallel(n_jobs=-1)(delayed(_evaluate_ssd_config)(p, X, ground_truth_labels, mask_all_meaningful_gt) for p in ssd_grid)
    best_ari_ssd, best_params_ssd = max(results_ssd, key=lambda item: item[0])
    print(f"Best SSDBSCAN params: {best_params_ssd} (ARI: {best_ari_ssd:.4f})")
    
    # C-DBSCAN Optimization
    print("\n--- Optimizing C-DBSCAN (Constraints)... ---")
    cdb_grid = [{'eps': eps, 'min_pts': ms} for eps in [0.17, 0.19, 0.24] for ms in [16, 24, 50]]
    results_cdb = Parallel(n_jobs=-1)(delayed(_evaluate_cdb_config)(p, X, must_link, cannot_link, ground_truth_labels, mask_all_meaningful_gt) for p in cdb_grid)
    best_ari_cdb, best_params_cdb = max(results_cdb, key=lambda item: item[0])
    print(f"Best C-DBSCAN params: {best_params_cdb} (ARI: {best_ari_cdb:.4f})")

    return {
        "climb": {'phase1': best_params_p1, 'phase2': best_params_p2},
        "ssdscan": best_params_ssd,
        "cdbscan": best_params_cdb
    }