import numpy as np
import itertools
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

class CDBSCAN:
    """
    C-DBSCAN (Constrained DBSCAN) clustering algorithm.
    This implementation supports Must-Link and Cannot-Link constraints.
    The implementation is based on the paper called C-DBSCAN: Density-Based Clustering with Constraints
    https://link.springer.com/chapter/10.1007/978-3-540-72530-5_25 
    """

    def __init__(self, eps=0.5, min_pts=5):
        """
        Initializes the C-DBSCAN model.

        Args:
            eps (float): The maximum distance between two samples for one to be considered
                         as in the neighborhood of the other.
            min_pts (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.
        """
        self.eps = eps
        self.min_pts = min_pts
        self.ml_set = set()
        self.cl_set = set()
        self.X = None
        self.n_samples = 0
        self.labels_ = None

    def _preprocess_constraints(self, ml, cl):
        """Preprocesses constraints into frozensets for efficient lookup."""
        self.ml_set = {frozenset(l) for l in ml}
        self.cl_set = {frozenset(l) for l in cl}

    def _check_cl(self, indices):
        """Checks if any pair of indices violates a Cannot-Link constraint."""
        return any(frozenset(pair) in self.cl_set for pair in itertools.combinations(indices, 2))

    def _create_local(self):
        """
        Creates local micro-clusters.
        
        This version protects points involved in Must-Link constraints from being
        prematurely labeled as noise, ensuring they can form micro-clusters
        even if they are in sparse regions.
        """
        print("  Step 2: Creating local clusters...")
        labels = np.full(self.n_samples, -2, dtype=int)  # -2: unvisited
        current_cluster_id = 0
        kdtree = KDTree(self.X)

        protected_points = set()
        for p1, p2 in self.ml_set:
            protected_points.add(p1)
            protected_points.add(p2)

        for i in range(self.n_samples):
            if labels[i] != -2:  # Skip if already visited
                continue

            neighbors_indices = kdtree.query_radius([self.X[i]], r=self.eps)[0]
            is_dense = len(neighbors_indices) >= self.min_pts
            is_protected = i in protected_points

            if not is_dense and not is_protected:
                labels[i] = -1  # Label as noise ONLY if not dense AND not protected
                continue

            # If the point survives, proceed to form micro-clusters.
            # The logic here handles Cannot-Link constraints and assigns cluster IDs.
            if self._check_cl(neighbors_indices):
                # If neighbors have a Cannot-Link, create a separate micro-cluster for each.
                for neighbor_idx in neighbors_indices:
                    if labels[neighbor_idx] == -2:
                        labels[neighbor_idx] = current_cluster_id
                        current_cluster_id += 1
            else:
                # If the neighborhood is "safe", assign all unvisited neighbors to the same new micro-cluster.
                # First, check if any neighbor already has a cluster. If so, join it.
                existing_labels = {labels[n_idx] for n_idx in neighbors_indices if labels[n_idx] >= 0}
                if existing_labels:
                    # Join the first existing cluster found
                    assign_id = min(existing_labels)
                else:
                    # Otherwise, create a new cluster ID
                    assign_id = current_cluster_id
                    current_cluster_id += 1
                
                # Assign the ID to all unvisited or noise points in the neighborhood
                for neighbor_idx in neighbors_indices:
                    if labels[neighbor_idx] in [-2, -1]:
                        labels[neighbor_idx] = assign_id
        
        return labels

    def _merge_ml(self, labels):
        """Merges clusters based on Must-Link constraints."""
        for p1, p2 in self.ml_set:
            l1, l2 = labels[p1], labels[p2]
            if l1 != l2 and l1 != -1 and l2 != -1:
                labels[labels == l2] = l1
        return labels

    def _merge_closest(self, labels):
        """Merges the closest clusters if no Cannot-Link constraints are violated."""
        while True:
            unique_clusters = np.unique(labels[labels != -1])
            if len(unique_clusters) <= 1:
                break

            min_dist = np.inf
            pair_to_merge = None

            for i, j in itertools.combinations(range(len(unique_clusters)), 2):
                c1_id, c2_id = unique_clusters[i], unique_clusters[j]
                c1_indices = np.where(labels == c1_id)[0]
                c2_indices = np.where(labels == c2_id)[0]

                dist = np.min(cdist(self.X[c1_indices], self.X[c2_indices]))
                if dist < min_dist:
                    min_dist = dist
                    pair_to_merge = (c1_id, c2_id)

            if pair_to_merge is None or min_dist > self.eps:
                break

            c1_id, c2_id = pair_to_merge
            c1_indices = np.where(labels == c1_id)[0]
            c2_indices = np.where(labels == c2_id)[0]

            violation = any(frozenset({p1, p2}) in self.cl_set for p1 in c1_indices for p2 in c2_indices)
            if not violation:
                labels[labels == c2_id] = c1_id
            else:
                break
        return labels

    def _finalize(self, labels):
        """Renames clusters to be 0-indexed, with -1 for noise."""
        unique = np.unique(labels[labels != -1])
        label_map = {old: new for new, old in enumerate(unique)}
        return np.array([label_map.get(l, -1) for l in labels])

    def fit(self, X, must_link=None, cannot_link=None):
        """
        Fits the C-DBSCAN model to the data with given constraints.

        Args:
            X (np.ndarray): The input data.
            must_link (list, optional): List of Must-Link constraint pairs. Defaults to [].
            cannot_link (list, optional): List of Cannot-Link constraint pairs. Defaults to [].

        Returns:
            self: The fitted C-DBSCAN instance.
        """
        print("\n--- [3/3] Executing C-DBSCAN (Constraints)... ---")
        if cannot_link is None:
            cannot_link = []
        if must_link is None:
            must_link = []

        self.X, self.n_samples = np.asarray(X), X.shape[0]
        self._preprocess_constraints(must_link, cannot_link)
        l1 = self._create_local()
        l2 = self._merge_ml(l1)
        l3 = self._merge_closest(l2)
        self.labels_ = self._finalize(l3)
        return self