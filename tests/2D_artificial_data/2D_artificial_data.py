# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from sklearn import datasets
from sklearn.cluster import KMeans

import sys
sys.path.append("../../src/")
from CKMeans import CKMeans, InitMode
from frequencies import draw_frequencies, FrequencyType
from sketching import Sk
from utils import SSE, compute_scores, predict_labels

warnings.filterwarnings(action="ignore")
plt.style.use("bmh")
np.random.seed(42)
random.seed(42)

n = 100

data_blobs, labels_blobs = datasets.make_blobs(n_samples=n, centers=3, random_state=42)
data_moons, labels_moons = datasets.make_moons(n_samples=n, noise=0.05, random_state=42)
data_circles, labels_circles = datasets.make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=42)

all_datasets = [(data_blobs, labels_blobs), (data_moons, labels_moons), (data_circles, labels_circles)]

def plot_datasets(datasets):
    n = len(datasets)
    plt.figure(figsize=(5 * n, 4))
    for i, (X, true_labels) in enumerate(datasets, 1):
        plt.subplot(1, n, i)
        plt.scatter(X[:, 0], X[:, 1], c=true_labels, edgecolor="k", s=20)
        plt.title(f"Dataset {i}")
    plt.savefig(f"2D_artificial_data_{n}.png")

plot_datasets(all_datasets)

select_datasets = [all_datasets[0], all_datasets[2]]
plot_datasets(select_datasets)


def kmeans_2D(datasets):
    n = len(datasets)
    plt.figure(figsize=(5 * n, 4))
    for i, (X, true_labels) in enumerate(datasets, 1):
        kmeans = KMeans(len(np.unique(true_labels)), n_init=1)
        kmeans.fit(X)
        C = kmeans.cluster_centers_
        predicted_labels = kmeans.labels_

        scores = compute_scores(true_labels, predicted_labels)
        scores["SSE/N"] = kmeans.inertia_/len(X)
        rand_index = scores["ARI"]

        print(f"Dataset {i}")
        print("".join(f"{metric_name + ' '*(8-len(metric_name))}\t: {value:.2f}\n" for metric_name, value in scores.items()))
        plt.subplot(1, n, i)
        plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k', s=20)
        plt.scatter(C[:, 0], C[:, 1], c='red', marker='o', s=50, label='Centroids')
        plt.title(f"Dataset {i} \nAdjusted Rand Index: {rand_index:.2f}\nSSE/N: {(kmeans.inertia_/len(X)):.2f}")
        plt.legend()

    plt.savefig(f"2D_artificial_data_k_means{n}.png")

kmeans_2D(all_datasets)

kmeans_2D(select_datasets)


plt.figure(figsize=(15, 4))
for i, (X, true_labels) in enumerate(all_datasets, 1):
    d = 2
    K = len(np.unique(true_labels))
    N = len(X)
    l = np.min(X) * np.ones(d)
    u = np.max(X) * np.ones(d)

    # frequencies
    m = K * N
    m0 = m
    c = 10
    n0 = N // 10
    display = False

    R = np.random.random(c)
    T = N // 10
    Omega = draw_frequencies(X, m, n0, m0, c, T, R, FrequencyType.GAUSSIAN, display)

    # sketching
    z = Sk(Omega, X)

    # CKMeans
    C, alpha = CKMeans(z, Omega, K, l, u, X, InitMode.RANGE, display)
    predicted_labels = predict_labels(X, C)

    sse = SSE(X, C) / N
    scores = compute_scores(true_labels, predicted_labels)
    scores["SSE/N"] = sse
    rand_index = scores["ARI"]

    print(f"Dataset {i}")
    print("".join(f"{metric_name + ' '*(8-len(metric_name))}\t: {value:.2f}\n" for metric_name, value in scores.items()))
    plt.subplot(1, 3, i)
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k', s=20)
    plt.scatter(C[:, 0], C[:, 1], c="red", marker='o', s=50)
    plt.title(f"Dataset {i} \nAdjusted Rand Index: {rand_index:.2f}\nSSE/N: {sse:.2f}")

plt.savefig(f"2D_artificial_data_CKM_3.png")


N = 10_000
X, true_labels = datasets.make_blobs(n_samples=N, centers=3, random_state=42)

colors = list(map(lambda i: ["#48cae4", "#2ecc71", "#ff7b94"][i], true_labels))

plt.figure(figsize=(5, 4))
plt.subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=colors, s=3, edgecolors=colors)
plt.title(f"Dataset N={N}")
plt.savefig(f"2D_artificial_data_{10_000}.png")

d = 2
K = 3

l = np.min(X) * np.ones(d)
u = np.max(X) * np.ones(d)

m_factor_list = np.arange(1, 5)

all_scores                 = {}
all_scores["ARI"]          = []
all_scores["AMI"]          = []
all_scores["homogeneity"]  = []
all_scores["completeness"] = []
all_scores["v-measure"]    = []
all_scores["SSE/N"]        = []

for m_factor in m_factor_list:
    # frequencies
    m = m_factor * K * N
    m0 = m
    n0 = N // 100
    display = False
    c = m // 10
    R = np.random.random(c)
    T = 10
    Sigma = 0.5 * np.identity(d)
    Omega = draw_frequencies(X, m, n0, m0, c, T, R, FrequencyType.GAUSSIAN, display)

    # sketching
    z = Sk(Omega, X)

    # CKMeans
    C, alpha = CKMeans(z, Omega, K, l, u, X, InitMode.RANGE, display)
    predicted_labels = predict_labels(X, C)

    sse = SSE(X, C) / N
    scores = compute_scores(true_labels, predicted_labels)
    scores["SSE/N"] = sse

    for score_name, score in scores.items():
        all_scores[score_name].append(score)

plt.figure(figsize=(15, 10))
for i, (score_name, scores) in enumerate(all_scores.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(m_factor_list, scores)
    plt.title(score_name)
plt.savefig(f"2D_artificial_data_CKM_grid_search_m_factor.png")

plt.figure(figsize=(10, 4))
for (score_name, scores) in all_scores.items():
    if score_name == "SSE/N": continue
    plt.subplot(1, 2, 1)
    plt.plot(m_factor_list, scores, label=score_name)
    plt.xticks(m_factor_list)
    plt.legend()
plt.subplot(1, 2, 2)
plt.plot(m_factor_list, all_scores["SSE/N"], label="SSE/N")
plt.xticks(m_factor_list)
plt.legend()
plt.savefig(f"2D_artificial_data_CKM_grid_search_m_factor_2.png")

print(all_scores["SSE/N"])
m_factor = np.argmin(all_scores["SSE/N"]) + 1

d = 2
K = 3

l = np.min(X) * np.ones(d)
u = np.max(X) * np.ones(d)

# frequencies
m = m_factor * K * N
m0 = m
n0 = N // 100
display = False
c = m // 10
R = np.random.random(c)
T = 10
Sigma = 0.5 * np.identity(d)
Omega = draw_frequencies(X, m, n0, m0, c, T, R, FrequencyType.GAUSSIAN, display)

# sketching
z = Sk(Omega, X)

# CKMeans
C, alpha = CKMeans(z, Omega, K, l, u, X, InitMode.RANGE, display)
predicted_labels = predict_labels(X, C)

sse = SSE(X, C) / N
scores = compute_scores(true_labels, predicted_labels)
scores["SSE/N"] = sse
rand_index = scores["ARI"]

print("".join(f"{metric_name + ' '*(8-len(metric_name))}\t: {value:.2f}\n" for metric_name, value in scores.items()))
plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels,cmap='viridis', edgecolor='k', s=20)
plt.scatter(C[:, 0], C[:, 1], c="red", marker='o', s=50)
plt.title(f"\nAdjusted Rand Index: {rand_index:.2f}\nSSE/N: {sse:.2f}")
plt.savefig(f"2D_artificial_data_CKM_best_m_factor.png")

print(C)
