# Astrophysical Clustering Analysis with CLiMB

## 1. Overview

This project provides a comprehensive pipeline for semi-supervised clustering of astrophysical data, specifically focusing on identifying stellar structures from datasets like RRLyrae stars. The core of the project is a comparative analysis between the **CLiMB** algorithm, a constrained version of DBSCAN (**C-DBSCAN**), and a baseline **SSDBSCAN** modified with an astrophysical heuristic.

The pipeline is designed to be modular, configurable, and easily extensible, handling everything from data preparation and algorithm execution to performance evaluation and advanced visualization.

## 2. Features

- **Three Clustering Algorithms**:
    - **CLiMB**: A semi-supervised algorithm that grows clusters from known seeds (Phase 1) and discovers new ones in the remaining data (Phase 2).
    - **C-DBSCAN**: A constrained version of DBSCAN that respects Must-Link and Cannot-Link constraints derived from seed points.
    - **SSDBSCAN with Heuristic**: A semi-supervised DBSCAN that uses an astrophysical heuristic to guide cluster expansion.
- **Comparative Analysis**: Automatically calculates and prints the Adjusted Rand Index (ARI) to compare the performance of all algorithms against ground truth labels.
- **Advanced Visualization**: Generates high-quality plots for in-depth analysis:
    - A detailed **diagnostic plot** for CLiMB, showing the results of its two phases separately.
    - A **comparison panel** that visualizes the results of all algorithms side-by-side with the ground truth.
    - Optional exclusion of noise points for clearer cluster visualization.
- **Seed Sensitivity Analysis**: Includes an optional module to automatically analyze and plot how the performance of the algorithms (ARI and Davies-Bouldin score) changes with the percentage of initial seed points provided.
- **Modular & Configurable**: The project is heavily refactored into logical modules (algorithms, plotting, utils), and its behavior can be easily controlled via switches in a central configuration file (`utils/constants.py`).

## 3. Project Structure

```
astrophysical-clustering/
│
├── algorithms/                 # Core implementations of clustering algorithms
│   ├── cdbscan.py              # C-DBSCAN class based on https://link.springer.com/chapter/10.1007/978-3-540-72530-5_25
│   └── ssd_heuristic.py        # SSDBSCAN with astrophysical heuristic functions 
│                               # based on https://github.com/TibaZaki/SS_DBSCAN
│
├── data/                       # main datataset
│   ├── ...
│
├── plots/                      # Visualization functions
│   
│
├── utils/                      # Utility scripts and configuration
│   ├── constants.py            # Global constants and configuration switches
│   ├── plots.py                # Functions for all generated plots (diagnostic, comparison, analysis)
│   └── utils.py                # Data loading, preprocessing, and evaluation helpers
│
├── known_clusters/             # Directory for input data (catalogue, centroids, etc.)
│   ├── ...
│
├── plots/                      # Default output directory for all generated plots
│
├── main.py                     # Main script to run the entire analysis pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # This file```

## 4. Installation

**Prerequisites**:
- Python 3.8+
- Git

**Steps**:

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd astrophysical-clustering
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## 5. Configuration

The main behavior of the script can be controlled by editing the boolean switches in `utils/constants.py`.

| Constant              | Type    | Default | Description                                                              |
| --------------------- | ------- | ------- | ------------------------------------------------------------------------ |
| `is_optimize`         | Boolean | `False` | If `True`, runs a lengthy hyperparameter optimization process.           |
| `preprocess_energy`   | Boolean | `False` | If `True`, applies the energy preprocessing step.                        |

## 6. Usage

To run the entire analysis pipeline, simply execute the `main.py` script from the root directory of the project:

```sh
python main.py
```

The script will:
1.  Load and preprocess the data.
2.  Run CLiMB, C-DBSCAN, and SSDBSCAN with the default or optimized parameters.
3.  Print the final ARI scores for each algorithm to the console.
4.  Generate and save the diagnostic, comparison and performance_vs_seed plots in the `./plots/` directory.

## 7. Dependencies

The project relies on the following Python libraries, which are listed in `requirements.txt`:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pyclustertend`
- `seaborn`
- `joblib`
- `climb-astro`