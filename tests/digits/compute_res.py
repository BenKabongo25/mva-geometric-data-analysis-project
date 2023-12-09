# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import json
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("bmh")
BASE_PATH = "Project/tests/digits/"


with open(BASE_PATH + "digits_all_K_Means_scores_1.json", "r") as f:
    all_K_Means_scores_1 = json.load(f)


with open(BASE_PATH + "digits_all_CKM_scores_1.json", "r") as f:
    all_CKM_scores_1 = json.load(f)


inits = ["RANGE", "SAMPLE", "KPP"] 

for score_name in all_K_Means_scores_1.keys():
    print(score_name)

    init_scores_K_Means = all_K_Means_scores_1[score_name]
    print("\tK-Means")
    for init_mode in inits:
        scores = init_scores_K_Means[init_mode]
        print(f"\t\t{init_mode} : {np.mean(scores)} ({np.std(scores)})")

    init_scores_CKM = all_CKM_scores_1[score_name]
    print("\tCKM")
    for init_mode in inits:
        scores = init_scores_CKM[init_mode]
        print(f"\t\t{init_mode} : {np.mean(scores)} ({np.std(scores)})")
