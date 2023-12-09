# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import json
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


import sys
sys.path.append("../../src/")
from CKMeans import CKMeans, InitMode
from frequencies import draw_frequencies, FrequencyType
from sketching import Sk
from utils import SSE, compute_scores, predict_labels


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


warnings.filterwarnings(action="ignore")
np.random.seed(42)
random.seed(42)


mnist = fetch_openml('mnist_784')
X = np.array(mnist.data)
Y = np.array(mnist.target)
K = len(np.unique(Y))
print(X.shape, Y.shape)
print(K)
plt.imshow(X[3333].reshape((28, 28)), cmap="gray")
plt.axis("off")
plt.savefig("mnist_example.png")


plt.style.use("bmh")


n_experimentations = 100


def mnist_kmeans(init="k-means++", n_init=1):
    kmeans = KMeans(K, init=init, n_init=n_init)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    predicted_labels = predict_labels(X, C)
    scores = compute_scores(Y, predicted_labels)
    scores["SSE/N"] = SSE(X, C) / N
    return scores


def kmeans_experimentaion1():
    all_K_Means_scores_1                 = {}
    all_K_Means_scores_1["ARI"]          = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_K_Means_scores_1["AMI"]          = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_K_Means_scores_1["homogeneity"]  = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_K_Means_scores_1["completeness"] = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_K_Means_scores_1["v-measure"]    = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_K_Means_scores_1["SSE/N"]        = {"RANGE": [], "SAMPLE": [], "KPP": []}

    for _ in tqdm(range(n_experimentations)):

        scores = mnist_kmeans(init="random", n_init=1)
        for score_name, score in scores.items():
            all_K_Means_scores_1[score_name]["RANGE"].append(score)
        
        C = X[np.random.choice(np.arange(0, len(X)), size=K, replace=True)]
        scores = mnist_kmeans(init=C, n_init=1)
        for score_name, score in scores.items():
            all_K_Means_scores_1[score_name]["SAMPLE"].append(score)

        scores = mnist_kmeans(init="k-means++", n_init=1)
        for score_name, score in scores.items():
            all_K_Means_scores_1[score_name]["KPP"].append(score)

        with open("mnist_all_K_Means_scores_1.json", "w") as f:
            txt = json.dumps(all_K_Means_scores_1)
            f.write(txt)

        for score_name, init_scores in all_K_Means_scores_1.items():
            data = np.array([init_scores["RANGE"], init_scores["SAMPLE"], init_scores["KPP"]]).T
            labels = ["RANGE", "SAMPLE", "KPP"]
            plt.figure(figsize=(5, 4))
            plt.title(f"K-Means {score_name}")
            plt.boxplot(data, labels=labels)
            score_name = score_name.replace("/", "_")
            plt.savefig(f"mnist_kmeans_exp1_{score_name}.png")

    return all_K_Means_scores_1


print("K-Means experimentation 1 ...")
all_K_Means_scores_1 = kmeans_experimentaion1()
with open("mnist_all_K_Means_scores_1.json", "r") as f:
    all_K_Means_scores_1 = json.load(f)


print("CKM ...")
N = len(X)
d = 784
l = np.min(X) * np.ones(d)
u = np.max(X) * np.ones(d)


def compute_frequencies():
    # frequencies
    m = K * N
    m0 = m
    c = 10
    n0 = N // 10
    R = np.random.random(c)
    T = 5
    Omega = draw_frequencies(X, m, n0, m0, c, T, R, FrequencyType.GAUSSIAN, False)
    np.savetxt("mnist_Omega.txt", Omega)
    return Omega

def load_frequencies():
    Omega = np.loadtxt("mnist_Omega.txt", dtype=float)
    return Omega


print("Compute frequencies ...")
Omega = compute_frequencies()
Omega = load_frequencies()


def compute_sketch():
    # sketching
    z = Sk(Omega, X)
    np.savetxt("mnist_sketch.txt", z)
    return z

def load_sketch():
    z = np.loadtxt("mnist_sketch.txt", dtype=np.complex128)
    return z


print("Compute sketch ...")
z = compute_sketch()
z = load_sketch()


def mnist_CKMeans(init):
    C, _ = CKMeans(z, Omega, K, l, u, X, init, False)
    predicted_labels = predict_labels(X, C)
    scores = compute_scores(Y, predicted_labels)
    scores["SSE/N"] = SSE(X, C) / N
    return scores


def CKM_experimentation1():
    all_CKM_scores_1                 = {}
    all_CKM_scores_1["ARI"]          = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_CKM_scores_1["AMI"]          = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_CKM_scores_1["homogeneity"]  = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_CKM_scores_1["completeness"] = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_CKM_scores_1["v-measure"]    = {"RANGE": [], "SAMPLE": [], "KPP": []}
    all_CKM_scores_1["SSE/N"]        = {"RANGE": [], "SAMPLE": [], "KPP": []}

    for _ in tqdm(range(n_experimentations), "Runs"):

        scores = mnist_CKMeans(init=InitMode.RANGE)
        for score_name, score in scores.items():
            all_CKM_scores_1[score_name]["RANGE"].append(score)
        
        scores = mnist_CKMeans(init=InitMode.SAMPLE)
        for score_name, score in scores.items():
            all_CKM_scores_1[score_name]["SAMPLE"].append(score)

        scores = mnist_CKMeans(init=InitMode.KPP)
        for score_name, score in scores.items():
            all_CKM_scores_1[score_name]["KPP"].append(score)

        with open("mnist_all_CKM_scores_1.json", "w") as f:
            txt = json.dumps(all_CKM_scores_1)
            f.write(txt)

        for score_name, init_scores in all_CKM_scores_1.items():
            data = np.array([init_scores["RANGE"], init_scores["SAMPLE"], init_scores["KPP"]]).T
            labels = ["RANGE", "SAMPLE", "KPP"]
            plt.figure(figsize=(5, 4))
            plt.title(f"CKM {score_name}")
            plt.boxplot(data, labels=labels)
            score_name = score_name.replace("/", "_")
            plt.savefig(f"mnist_ckm_exp1_{score_name}.png")


print("CKM experimentation 1 ...")
all_CKM_scores_1 = CKM_experimentation1()
with open("mnist_all_CKM_scores_1.json", "r") as f:
    all_CKM_scores_1 = json.load(f)


for score_name in all_K_Means_scores_1.keys():
    init_scores_K_Means = all_K_Means_scores_1[score_name]
    init_scores_CKM = all_CKM_scores_1[score_name]
    data = np.array([init_scores_K_Means["RANGE"], init_scores_CKM["RANGE"],
                    init_scores_K_Means["SAMPLE"], init_scores_CKM["SAMPLE"],
                    init_scores_K_Means["KPP"], init_scores_CKM["KPP"]]).T
    labels = ["RANGE K-Means", "RANGE CKM", "SAMPLE K-Means", "SAMPLE CKM", "KPP K-Means", "KPP CKM"]
    plt.figure(figsize=(10, 4))
    plt.title(score_name)
    plt.boxplot(data, labels=labels)
    score_name = score_name.replace("/", "_")
    plt.savefig(f"mnist_all_exp1_{score_name}.png")
