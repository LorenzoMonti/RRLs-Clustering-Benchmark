import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score

from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory
from algorithms.ssd_heuristic import run_heuristic_ssdscan
from algorithms.cdbscan import CDBSCAN
from utils.utils import create_seeds_dict_with_indices

# --- Custom Legend Handler for Colormaps ---
class HandlerColormap(HandlerBase):
    """Custom legend handler to display a colormap swatch."""
    def __init__(self, cmap, num_stripes=10, **kw):
        super().__init__(**kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        stripe_width = width / self.num_stripes
        for i in range(self.num_stripes):
            stripe = Rectangle([xdescent + i * stripe_width, ydescent], stripe_width, height,
                               facecolor=self.cmap(i / (self.num_stripes - 1)),
                               edgecolor='none', transform=trans)
            stripes.append(stripe)
        frame = Rectangle([xdescent, ydescent], width, height,
                          facecolor='none', edgecolor='black', lw=0.5, transform=trans)
        stripes.append(frame)
        return stripes


def plot_climb_diagnostic(climb_instance, X_original, title, save_path, show_noise=True):
    """
    Creates a high-quality diagnostic plot for CLiMB results.

    Args:
        climb_instance: The fitted CLiMB object.
        X_original (np.ndarray): The original UNSCALED dataset.
        title (str): The title of the plot.
        save_path (str): The path to save the image.
        show_noise (bool, optional): If True, plots noise points. Defaults to True.
    """
    print(f"\n--- Generating CLiMB Diagnostic Plot: {title} ---")
    plt.figure(figsize=(14, 10))

    # --- 1. Isolate data ---
    constrained_mask = (climb_instance.constrained_labels != -1)
    constrained_indices = np.where(constrained_mask)[0]
    constrained_labels = climb_instance.constrained_labels[constrained_mask]

    unassigned_indices = np.where(climb_instance.constrained_labels == -1)[0]
    exploratory_labels = climb_instance.exploratory_labels
    exploratory_clustered_mask = (exploratory_labels != -1)
    discovered_indices = unassigned_indices[exploratory_clustered_mask]
    discovered_labels = exploratory_labels[exploratory_clustered_mask]
    final_noise_indices = unassigned_indices[(exploratory_labels == -1)]

    # --- 2. Plot layers ---
    if show_noise and len(final_noise_indices) > 0:
        plt.scatter(X_original[final_noise_indices, 0], X_original[final_noise_indices, 1],
                    c='gray', s=10, alpha=0.3)
    if len(constrained_indices) > 0:
        plt.scatter(X_original[constrained_indices, 0], X_original[constrained_indices, 1],
                    c=constrained_labels, cmap='viridis', s=20, ec='black', lw=0.2)
    if len(discovered_indices) > 0:
        plt.scatter(X_original[discovered_indices, 0], X_original[discovered_indices, 1],
                    c=discovered_labels, cmap='autumn', s=35, ec='black', lw=0.5)

    # --- 3. Formatting and Custom Legend ---
    plt.title(title, fontsize=18)
    plt.xlabel('Lz (10³ kpc km/s)', fontsize=14)
    plt.ylabel('E (10⁵ km/s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    handles = [Rectangle((0, 0), 1, 1), Rectangle((0, 0), 1, 1)]
    labels = ['Constrained Clusters (cool colors)', 'Discovered Clusters (warm colors)']
    handler_map = {handles[0]: HandlerColormap(plt.cm.viridis), handles[1]: HandlerColormap(plt.cm.autumn)}

    if show_noise:
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='gray', markersize=10, alpha=0.5, linestyle='None'))
        labels.append('Final Noise')

    plt.legend(handles=handles, labels=labels, handler_map=handler_map, loc='best', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_comparison_panel(X_original, ground_truth_labels, mask_known_substructures, mask_all_meaningful_gt, all_labels, save_path, show_noise=True):
    """
    Generates a 2x2 comparison panel.
    - Top-Left: Ground Truth displaying ONLY the KNOWN substructures (Phase 1 target).
    - Others: Algorithm results compared against ALL meaningful structures.
    """
    print(f"\n--- Generating Comparison Panel Plot ---")
    
    # Safety Check: Ensure mask is boolean and has correct shape
    if mask_known_substructures.shape[0] != X_original.shape[0]:
        raise ValueError(f"Mask shape {mask_known_substructures.shape} does not match Data shape {X_original.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 18), sharex=True, sharey=True)
    fig.suptitle('Comparison of Found Clusters', fontsize=22)
    
    true_labels_meaningful = ground_truth_labels[mask_all_meaningful_gt]

    # --- 1. Plot Ground Truth (subplot 0,0) ---
    ax_gt = axes[0, 0]
    
    if show_noise:
        mask_background = ~mask_known_substructures
        ax_gt.scatter(X_original[mask_background, 0], X_original[mask_background, 1], 
                      c='gray', s=10, alpha=0.2, label='Background/Unknown')
    
    # Only known clusters
    # ! We use mask_known_substructures to filter X and labels
    ax_gt.scatter(X_original[mask_known_substructures, 0], X_original[mask_known_substructures, 1],
                  c=ground_truth_labels[mask_known_substructures], cmap='viridis', s=20, ec='black', lw=0.2)
    
    ax_gt.set_title('Ground Truth (Known Substructures Only)', fontsize=16)
    ax_gt.set_ylabel('E (10⁵ km/s)', fontsize=14)

    # --- 2. Plot Algorithm Results ---
    ax_flat = [axes[0, 1], axes[1, 0], axes[1, 1]]
    
    items = list(all_labels.items())
    if len(items) > 3: 
        items = items[:3]

    for i, (name, labels) in enumerate(items):
        ax = ax_flat[i]
        
        clustered_mask = (labels != -1)
        noise_mask = ~clustered_mask
        
        ari = adjusted_rand_score(true_labels_meaningful, labels[mask_all_meaningful_gt])
        title = f'{name} - ARI: {ari:.3f}'
        
        if show_noise:
            ax.scatter(X_original[noise_mask, 0], X_original[noise_mask, 1], c='gray', s=10, alpha=0.2)
        
        ax.scatter(X_original[clustered_mask, 0], X_original[clustered_mask, 1],
                   c=labels[clustered_mask], cmap='viridis', s=20, ec='black', lw=0.2)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Lz (10³ kpc km/s)', fontsize=14)
        if i == 1: ax.set_ylabel('E (10⁵ km/s)', fontsize=14)
            
    # --- 3. Final Formatting ---
    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison panel saved to {save_path}")


def plot_performance_vs_seeds(X, ground_truth_labels, mask_all_meaningful_gt, rrl_dataset, scaler, optimal_params, known_labels, save_dir):
    """
    Analyzes and plots ARI and Davies-Bouldin scores as a function of seed percentage.

    This function iterates through different seed percentages, re-runs all clustering
    algorithms for each percentage, calculates performance metrics, and plots the results.
    
    Args:
        X (np.ndarray): The scaled dataset.
        ground_truth_labels (np.ndarray): True labels for ARI calculation.
        mask_all_meaningful_gt (np.ndarray): Mask for meaningful GT labels.
        rrl_dataset (np.ndarray): The original, unscaled dataset for seed generation.
        scaler (StandardScaler): The fitted scaler object.
        optimal_params (dict): Dictionary with optimal hyperparameters for the algorithms.
        known_labels (np.ndarray): Array of known cluster labels.
        save_dir (str): Directory to save the output plots.
    """
    print("\n--- [ANALYSIS] Running Performance vs. Seed Percentage Analysis ---")
    
    seed_percentages = np.arange(10, 101, 10)  # 10, 20, 30, ..., 100
    ari_scores = {"CLiMB": [], "C-DBSCAN": [], "SSDBSCAN": []}
    db_scores_climb = []
    
    true_labels_meaningful = ground_truth_labels[mask_all_meaningful_gt]

    for p in seed_percentages:
        print(f"  Testing with seed_percentage = {p}%")
        
        # 1. Generate seeds and constraints for the current percentage
        seeds_info, _ = create_seeds_dict_with_indices(
            "known_clusters/output/", "known_clusters/clusters/",
            rrl_dataset, seed_percentage=p, scaler=scaler
        )
        seeds_for_climb = {tuple(info['centroid_coords_scaled']): [tuple(c) for c in info['seed_coords_scaled']] for _, info in seeds_info.items()}
        must_link = [item for info in seeds_info.values() for item in itertools.combinations(info['seed_indices'], 2)]
        indices_list = [v['seed_indices'] for v in seeds_info.values()]
        cannot_link = [(indices_list[i][0], indices_list[j][0]) for i in range(len(indices_list)) for j in range(i + 1, len(indices_list)) if len(indices_list[i]) > 0 and len(indices_list[j]) > 0]

        # 2. Run algorithms and calculate metrics
        # CLiMB
        climb = CLiMB(
            constrained_clusters=len(seeds_for_climb), seed_points=seeds_for_climb, **optimal_params['climb']['phase1'],
            distance_metric="mahalanobis", metric_params={'VI': np.linalg.inv(np.cov(X.T))},
            exploratory_algorithm=DBSCANExploratory(**optimal_params['climb']['phase2'])
        )
        climb.fit(X, known_labels, is_slight_movement=False)
        final_climb_labels = np.copy(climb.constrained_labels)
        unassigned_indices = np.where(final_climb_labels == -1)[0]
        final_climb_labels[unassigned_indices] = climb.exploratory_labels
        
        ari_scores["CLiMB"].append(adjusted_rand_score(true_labels_meaningful, final_climb_labels[mask_all_meaningful_gt]))
        
        # Davies-Bouldin for CLiMB
        climb_clustered_mask = (final_climb_labels != -1)
        if len(np.unique(final_climb_labels[climb_clustered_mask])) > 1:
            db_score = davies_bouldin_score(X[climb_clustered_mask], final_climb_labels[climb_clustered_mask])
            db_scores_climb.append(db_score)
        else:
            db_scores_climb.append(np.nan) # Append NaN if score is not computable

        # C-DBSCAN
        cdbscan = CDBSCAN(**optimal_params['cdbscan'])
        cdbscan.fit(X, must_link=must_link, cannot_link=cannot_link)
        ari_scores["C-DBSCAN"].append(adjusted_rand_score(true_labels_meaningful, cdbscan.labels_[mask_all_meaningful_gt]))
        
        # SSDBSCAN (independent of seeds, but run for consistency)
        ssd_labels = run_heuristic_ssdscan(X, **optimal_params['ssdscan'])
        ari_scores["SSDBSCAN"].append(adjusted_rand_score(true_labels_meaningful, ssd_labels[mask_all_meaningful_gt]))

    # 3. Plot ARI results
    plt.figure(figsize=(12, 8))
    plt.plot(seed_percentages, ari_scores["CLiMB"], 'o-', label="CLiMB")
    plt.plot(seed_percentages, ari_scores["C-DBSCAN"], 's-', label="C-DBSCAN")
    # SSDBSCAN is constant, plot as a dashed line
    plt.axhline(y=ari_scores["SSDBSCAN"][0], color='r', linestyle='--', label="SSDBSCAN (Constant)")
    
    plt.title("Sensitivity analysis", fontsize=16)
    plt.xlabel("Seed Percentage (%)", fontsize=12)
    plt.ylabel("Adjusted Rand Index (ARI)", fontsize=12)
    plt.xticks(seed_percentages)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"{save_dir}/ari_vs_seeds_analysis.png", dpi=300)

    # 4. Plot Davies-Bouldin results for CLiMB
    plt.figure(figsize=(12, 8))
    plt.plot(seed_percentages, db_scores_climb, '^-', color='green', label="CLiMB Davies-Bouldin Score")
    
    plt.title("CLiMB Davies-Bouldin Score vs. Seed Percentage", fontsize=16)
    plt.xlabel("Seed Percentage (%)", fontsize=12)
    plt.ylabel("Davies-Bouldin Score (Lower is Better)", fontsize=12)
    plt.xticks(seed_percentages)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"{save_dir}/db_score_vs_seeds_analysis.png", dpi=300)
