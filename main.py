import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, precision_score, recall_score
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
    dataset = select_features(rrl_dataset, [2, 1, 3])  # Lz, Energy, Lperp
    ground_truth_labels = np.asarray(select_features(rrl_dataset, [4])).flatten()
    
    # Preprocessing
    if preprocess_energy:
        dataset = energy_preprocessing(dataset, times=1000)
    scaler = StandardScaler()
    X = scaler.fit_transform(dataset)
    
    # Seed and Constraint Generation
    seeds_info, _ = create_seeds_dict_with_indices(
        "known_clusters/output/", "known_clusters/clusters/",
        rrl_dataset, seed_percentage=90, scaler=scaler
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
    
    # --- [PHASE 2] ALGORITHM EXECUTION ---
    print("\n\n--- [PHASE 2] Algorithm Execution ---")
    print("\n--- [1/3] Executing CLiMB... ---")
    climb = CLiMB(
        constrained_clusters=len(seeds_for_climb), seed_points=seeds_for_climb,
        **optimal_params['climb']['phase1'],
        distance_metric="mahalanobis", metric_params={'VI': np.linalg.inv(np.cov(X.T))},
        exploratory_algorithm=DBSCANExploratory(**optimal_params['climb']['phase2'])
    )
    climb.fit(X, None, is_slight_movement=False)
    final_climb_labels = np.copy(climb.constrained_labels)
    unassigned_indices = np.where(final_climb_labels == -1)[0]
    final_climb_labels[unassigned_indices] = climb.exploratory_labels

    # 2. Heuristic SSDBSCAN Execution
    heuristic_ssdscan_labels = run_heuristic_ssdscan(X, **optimal_params['ssdscan'])
    
    # 3. C-DBSCAN Execution
    cdbscan = CDBSCAN(**optimal_params['cdbscan'])
    cdbscan.fit(X, must_link=must_link, cannot_link=cannot_link)

    # --- [PHASE 3] COMPARATIVE EVALUATION ---
    print("\n\n--- [PHASE 3] Comparative Evaluation ---")

    # --- A. Prepare Masks and Data for Evaluation ---
    known_cluster_ids_provided = np.unique(centroids_dataset[:, 0]).astype(int)
    unclassified_id = 0

    mask_known_points = np.isin(ground_truth_labels, known_cluster_ids_provided)
    mask_unclassified = (ground_truth_labels == unclassified_id)
    mask_all_meaningful_gt = (ground_truth_labels != unclassified_id)

    all_labels = {
        "CLiMB": final_climb_labels,
        "SSDBSCAN": heuristic_ssdscan_labels,
        "C-DBSCAN": cdbscan.labels_
    }

    # --- Question 1: Knowledge Recovery Performance ---
    print("\n--- Evaluation 1: Knowledge Recovery and Baseline Comparison (on 8 known clusters) ---")
    true_labels_known = ground_truth_labels[mask_known_points]

    # Specific analysis for CLiMB's Phase 1
    print("\n- For CLiMB (Phase 1 only):")
    pred_labels_climb_p1 = climb.constrained_labels[mask_known_points]
    ari_p1 = adjusted_rand_score(true_labels_known, pred_labels_climb_p1)
    homog_p1 = homogeneity_score(true_labels_known, pred_labels_climb_p1)
    comp_p1 = completeness_score(true_labels_known, pred_labels_climb_p1)
    print(f"  ARI: {ari_p1:.4f} | Homogeneity: {homog_p1:.4f} | Completeness: {comp_p1:.4f}")
    
    print("\n--- Per-Cluster Metrics (Precision & Recall) per CLiMB Phase 1 ---")    
    # Finding unique labels of the ground truth (8 Dodd's clusters)
    unique_true_labels = np.unique(true_labels_known)
    
    for true_lbl in unique_true_labels:
        # Finding which predicted label overlaps the most with this true cluster
        mask_true = (true_labels_known == true_lbl)
        if np.sum(mask_true) == 0: continue
        
        # Most frequent predicted label for this true class
        pred_lbls_for_this_class = pred_labels_climb_p1[mask_true]
        best_pred_lbl = pd.Series(pred_lbls_for_this_class).mode()[0]
        
        # Precision and Recall for this match
        y_true_binary = (true_labels_known == true_lbl)
        y_pred_binary = (pred_labels_climb_p1 == best_pred_lbl)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        print(f"Cluster GT ID {true_lbl}: Precision (Purity) = {precision:.3f}, Recall (Completeness) = {recall:.3f}")

    # Analysis for other algorithms
    other_algos = {"SSDBSCAN": heuristic_ssdscan_labels, "C-DBSCAN": cdbscan.labels_}
    for name, labels in other_algos.items():
        print(f"\n- For {name}:")
        pred_labels = labels[mask_known_points]
        ari = adjusted_rand_score(true_labels_known, pred_labels)
        homog = homogeneity_score(true_labels_known, pred_labels)
        comp = completeness_score(true_labels_known, pred_labels)
        print(f"  ARI: {ari:.4f} | Homogeneity: {homog:.4f} | Completeness: {comp:.4f}")

    # --- C. Question 2: Novelty Discovery Performance ---
    print("\n--- Evaluation 2: Novelty Discovery ---")
    
    print("\n- Exploratory Discovery Performance (from Field Stars)")
    for name, labels in all_labels.items():
        mask_discovered = mask_unclassified & (labels != -1)
        if np.sum(mask_discovered) > 1:
            discovered_points = X[mask_discovered]
            discovered_labels = labels[mask_discovered]
            if len(np.unique(discovered_labels)) > 1:
                dbi = davies_bouldin_score(discovered_points, discovered_labels)
                print(f"  {name}: Davies-Bouldin Score = {dbi:.4f} (on {len(discovered_points)} points)")
            else:
                print(f"  {name}: Davies-Bouldin Score = N/A (only 1 new cluster found)")
        else:
            print(f"  {name}: Davies-Bouldin Score = N/A (no new clusters found)")

    # --- D. Question 3: Overall Performance & Sensitivity ---
    print("\n--- Evaluation 3: Overall Performance and Sensitivity to Supervision ---")
    
    print("\n- Global Performance")
    true_labels_meaningful = ground_truth_labels[mask_all_meaningful_gt]
    for name, labels in all_labels.items():
        ari = adjusted_rand_score(true_labels_meaningful, labels[mask_all_meaningful_gt])
        print(f"  Global ARI for {name}: {ari:.4f}")


    # --- RANDOM SEEDING ---
    print("\n--- Running Random Seeding Baseline ---")
    # Initialize CLiMB with k=8 and without seed_points (seed_points=None, random K-bound seeding)
    np.random.seed(1234)  # For reproducibility
    
    climb_random = CLiMB(
        constrained_clusters=8, 
        seed_points=None, 
        **optimal_params['climb']['phase1'],
        distance_metric="mahalanobis", metric_params={'VI': np.linalg.inv(np.cov(X.T))},
        exploratory_algorithm=DBSCANExploratory(**optimal_params['climb']['phase2'])
    )
    climb_random.fit(X, None, is_slight_movement=False)
    
    pred_labels_random = climb_random.constrained_labels[mask_known_points]
    ari_random = adjusted_rand_score(true_labels_known, pred_labels_random)
    print(f"CLiMB with Random Seeding - ARI: {ari_random:.4f}")

    # --- [PHASE 4] PLOTTING ---
    print("\n\n--- [PHASE 4] Generating Plots ---")
    X_original = scaler.inverse_transform(X)
    
    plot_climb_diagnostic(climb, X_original, 
        "CLiMB", 
        "./plots/CLiMB_diagnostic.png",
        show_noise=False
    )

    plot_labels = {
        "CLiMB (Phase 1)": climb.constrained_labels,
        "SSDBSCAN": heuristic_ssdscan_labels,
        "C-DBSCAN": cdbscan.labels_
    }

    plot_comparison_panel(
        X_original=X_original,
        ground_truth_labels=ground_truth_labels,
        mask_known_substructures=mask_known_points,
        mask_all_meaningful_gt=mask_all_meaningful_gt,
        all_labels=plot_labels,
        save_path="./plots/comparison_panel.png",
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

    