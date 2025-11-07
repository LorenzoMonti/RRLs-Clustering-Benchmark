"""
Constants.py
"""
import numpy as np


# PATHS
data_path = f"known_clusters/catalogue_all/catalogue_all_rrls.csv"
clusters_path = f"known_clusters/output/rrls_clusters.csv"

# BOOLEAN SWITCHES
preprocess_energy = False
is_optimize = False # do you want to optimize clusterings' hyperpameters?

dataset, result, centr_list = list(), list(), list()

# KNOWN CLUSTERS LABELS
known_labels = np.array([1, 2, 3, 4, 5, 6, 7, 12, 16])

OPTIMAL_PARAMS = {
            "climb": {'phase1': {'density_threshold': 0.005, 'distance_threshold': 0.5, 'radial_threshold': 0.1, 'convergence_tolerance': 0.01},
                      'phase2': {'eps': 0.190, 'min_samples': 24}},
            "ssdscan": {'eps': 0.15, 'min_samples': 16},
            "cdbscan": {'eps': 0.24, 'min_pts': 16}
        }
