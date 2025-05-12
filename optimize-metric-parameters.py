import numpy as np
import matplotlib.pyplot as plt
from language_distance_metrics import cltad_distance, l2v_distance
import multiprocessing as mp
import pickle
import scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

allScores=[
# fullfrompretrained below
"""
hy,cs,33.39
hy,ro,33.15
hy,ru,33.43
hy,hu,33.42
hy,tr,33.17
hy,hr,33.10
hy,ko,33.07
""",
"""
az,cs,23.50
az,ro,23.75
az,ru,23.76
az,hu,23.59
az,tr,23.96
az,hr,23.54
az,ko,23.67
""",
"""
ka,cs,31.12
ka,ro,31.40
ka,ru,31.23
ka,hu,31.25
ka,tr,31.09
ka,hr,31.27
ka,ko,30.72
""",
"""
be,cs,36.37
be,ro,35.61
be,ru,35.86
be,hu,35.56
be,tr,35.79
be,hr,35.63
be,ko,35.65
""",
"""
gl,cs,41.83
gl,ro,42.43
gl,ru,41.85
gl,hu,41.89
gl,tr,41.59
gl,hr,41.88
gl,ko,41.95
""",
#fullfromscratch8 below
"""
hy,cs,13.82
hy,ro,14.77
hy,ru,13.33
hy,hu,13.01
hy,tr,13.49
hy,hr,14.84
hy,ko,12.35
""",
"""
az,cs,4.01
az,ro,4.54
az,ru,4.08
az,hu,3.86
az,tr,3.96
az,hr,4.78
az,ko,3.29
""",
"""
ka,cs,9.30
ka,ro,10.30
ka,ru,9.20
ka,hu,10.51
ka,tr,9.05
ka,hr,9.81
ka,ko,8.13
""",
"""
be,cs,6.27
be,ro,6.20
be,ru,5.50
be,hu,5.90
be,tr,4.88
be,hr,6.29
be,ko,3.48
""",
"""
gl,cs,15.78
gl,ro,18.67
gl,ru,16.15
gl,hu,15.35
gl,tr,14.35
gl,hr,16.87
gl,ko,14.19
"""
]

