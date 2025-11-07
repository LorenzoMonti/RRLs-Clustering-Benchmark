import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
import itertools

# Import refactored modules
from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory
from algorithms.ssd_heuristic import run_heuristic_ssdscan
from algorithms.cdbscan import CDBSCAN
from utils.utils import (read_csv_dataset, read_csv_centroids, select_features,
                         energy_preprocessing, create_seeds_dict_with_indices, optimize_benchmarks)
from utils.constants import OPTIMAL_PARAMS, data_path, clusters_path, preprocess_energy, is_optimize, known_labels
from utils.plots import *


if __name__ == '__main__':
    print("--- [PHASE 1] Data Preparation ---")
    
    # Data Loading
    rrl_dataset, _ = read_csv_dataset(data_path)
    centroids_dataset = read_csv_centroids(clusters_path)
    dataset = select_features(rrl_dataset, [2, 1, 3])  # Lz, Energy, FeH
    ground_truth_labels = np.asarray(select_features(rrl_dataset, [4])).flatten()
    
    # Preprocessing
    if preprocess_energy:
        dataset = energy_preprocessing(dataset, times=1000)
    scaler = StandardScaler()
    X = scaler.fit_transform(dataset)
    
    # Seed and Constraint Generation
    seeds_info, _ = create_seeds_dict_with_indices(
        "known_clusters/output/", "known_clusters/clusters/",
        rrl_dataset, seed_percentage=30, scaler=scaler
    )
    seeds_for_climb = {tuple(info['centroid_coords_scaled']): [tuple(c) for c in info['seed_coords_scaled']] for _, info in seeds_info.items()}
    must_link = [list(itertools.combinations(info['seed_indices'], 2)) for info in seeds_info.values()]
    must_link = [item for sublist in must_link for item in sublist] # Flatten list
    
    indices_list = [v['seed_indices'] for v in seeds_info.values()]
    cannot_link = []
    for i in range(len(indices_list)):
        for j in range(i + 1, len(indices_list)):
            if len(indices_list[i]) > 0 and len(indices_list[j]) > 0:
                cannot_link.append((indices_list[i][0], indices_list[j][0]))

    # Default or optimized parameters
    if is_optimize:
        optimal_params = optimize_benchmarks(X, ground_truth_labels, centroids_dataset, must_link, cannot_link, seeds_for_climb, known_labels)
    else:
        optimal_params = OPTIMAL_PARAMS
    
    # --- Algorithm Execution ---
    print("\n--- [1/3] Executing CLiMB... ---")
    climb = CLiMB(
        constrained_clusters=len(seeds_for_climb), seed_points=seeds_for_climb,
        **optimal_params['climb']['phase1'],
        distance_metric="mahalanobis", metric_params={'VI': np.linalg.inv(np.cov(X.T))},
        exploratory_algorithm=DBSCANExploratory(**optimal_params['climb']['phase2'])
    )
    climb.fit(X, known_labels, is_slight_movement=False)
    final_climb_labels = np.copy(climb.constrained_labels)
    unassigned_indices = np.where(final_climb_labels == -1)[0]
    final_climb_labels[unassigned_indices] = climb.exploratory_labels

    heuristic_ssdscan_labels = run_heuristic_ssdscan(X, **optimal_params['ssdscan'])
    
    cdbscan = CDBSCAN(**optimal_params['cdbscan'])
    cdbscan.fit(X, must_link=must_link, cannot_link=cannot_link)
    cdbscan_labels = cdbscan.labels_

    # --- Evaluation ---
    print("\n\n--- [PHASE 2] Comparative Evaluation ---")
    mask_all_meaningful_gt = (ground_truth_labels != 0)
    true_labels_meaningful = ground_truth_labels[mask_all_meaningful_gt]
    
    all_labels = {"CLiMB": final_climb_labels, "SSDBSCAN": heuristic_ssdscan_labels, "C-DBSCAN": cdbscan_labels}
    
    for name, labels in all_labels.items():
        ari = adjusted_rand_score(true_labels_meaningful, labels[mask_all_meaningful_gt])
        print(f"  Global Meaningful ARI for {name}: {ari:.4f}")

    # --- Plotting ---
    print("\n--- [PHASE 3] Generating Plots ---")
    X_original = scaler.inverse_transform(X)
    
    plot_climb_diagnostic(climb, X_original, 
        "CLiMB Diagnostic Plot", 
        "./plots/CLiMB_diagnostic.png",
        show_noise=False
    )
    
    plot_comparison_panel(
        X_original=X_original,
        ground_truth_labels=ground_truth_labels,
        mask_all_meaningful_gt=mask_all_meaningful_gt,
        all_labels=all_labels,
        save_path="./plots/master_comparison_panel.png",
        show_noise=False
    )

    plot_performance_vs_seeds(
            X=X,
            ground_truth_labels=ground_truth_labels,
            mask_all_meaningful_gt=mask_all_meaningful_gt,
            rrl_dataset=rrl_dataset,
            scaler=scaler,
            optimal_params=optimal_params,
            known_labels=known_labels,
            save_dir="./plots"
        )