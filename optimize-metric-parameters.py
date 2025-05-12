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

