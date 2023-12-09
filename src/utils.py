# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics


def SSE(X: np.ndarray, C: np.ndarray) -> float:
    """Compute SSE
    :param X: data
    :param C: centroids
    :return SSE
    """
    return cdist(X, C).min(axis=1).sum()


def predict_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Predicts the label of examples based on the nearest centroid
    :param X: data
    :param C: centroids
    :return labels of data
    """
    return cdist(X, C).argmin(axis=1)


clustering_metrics = {
    "ARI": metrics.adjusted_rand_score,
    "AMI": metrics.adjusted_mutual_info_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "v-measure": metrics.v_measure_score,
}


def compute_scores(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict:
    """Compute different clustering scores"""
    results = {}
    for metric_name, metric_func in clustering_metrics.items():
        results[metric_name] = metric_func(true_labels, predicted_labels)
    return results

