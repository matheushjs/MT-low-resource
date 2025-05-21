import transformers, torch
import gc
import random
import numpy as np
import pandas as pd
import argparse, sys, os, re, time, pickle
import multiprocessing as mp
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModel, AutoConfig
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy import stats
from language_distance_metrics import cltad_distance

np.set_printoptions(linewidth=160)

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

langs1 = ["hy", "az", "ka", "be", "gl"]
langs2 = ["cs", "ro", "ru", "hu", "tr", "ko", "fr", "es", "ar", "he", "ja", "hr"]

args = []
for lang1 in langs1:
    for lang2 in langs2:
        args.append([lang1, lang2])

def func(args):
    return args[0], args[1], cltad_distance(args[0], args[1])

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    with mp.Pool(8) as p:
        results = p.map(func, args, chunksize=1)
    #results = calculate_dist(args[0])

    N = len(langs1) + len(langs2)
    mat = -np.ones((N, N))
    df = pd.DataFrame(mat, columns=langs1+langs2, index=langs1+langs2)
    for r in results:
        i, j, dist = r
        df.loc[i,j] = dist
        df.loc[j,i] = dist

    df.to_csv("cltad-distances-reduced.csv")
