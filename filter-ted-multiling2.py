import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys, os, re, time, unicodedata, pickle, html, copy, shutil
from iso639 import Language
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

dataset = load_dataset("json", data_files={
        "train": "./ted-multiling-filtered/train.json",
        "test": "./ted-multiling-filtered/test.json",
        "dev": "./ted-multiling-filtered/dev.json"
    })

# train_df = dataset["train"].to_dict()
# test_df = dataset["test"].to_dict()
# dev_df = dataset["dev"].to_dict()

def fix_punctuation(text):
    # Fixes "didn 't" -> "didn't"
    text = re.sub(r"(\w)\s+'(\w)", r"\1'\2", text)

    # Fixes "n' avons" -> "n'avons"
    text = re.sub(r"(\w)'\s+(\w)", r"\1'\2", text)

    # Fixes "n ' avons" -> "n'avons"
    text = re.sub(r"(\w) ' (\w)", r"\1'\2", text)

    # Changes ' ' (text) ' ' into "(text)"
    text = re.sub(r"' '\s+([^']*?)\s+' '", r'"\1"', text)

    # Changes " " (text) " " into "(text)"
    text = re.sub(r"\" \"\s+([^\"]*?)\s+\" \"", r'"\1"', text)

    # Changes " (text) " into "(text)"
    text = re.sub(r"\"\s+([^']*?)\s+\"", r'"\1"', text)

    # Removes space before punctuation
    text = re.sub(r"\s+([.,!?;:、。！？：；،)։\]])", r'\1', text)

    return text

def func(row):
    for k in row.keys():
        text = fix_punctuation(row[k])
        if text != row[k]:
            #print(f"Changed '{row[k]}' to '{text}'.")
            pass
        row[k] = text
    return row

train_dataset = dataset["train"].map(func, num_proc=4)
test_dataset = dataset["test"].map(func, num_proc=4)
dev_dataset = dataset["dev"].map(func, num_proc=4)

# for k in train_df.keys():
#     if k == "en":
#         continue
#     print(k)

#     data = pklload(f"laser/{k}.pickle")

#     emb = data["train"]["emb"]
#     idx = data["train"]["idx"]
#     emb_en = data_en["train"]["emb"]
#     df = train_df

#     sims = []
#     for i in range(emb.shape[0]):
#         emb1 = emb[i,:]
#         emb2 = emb_en[idx[i],:]
#         sim = cosine_similarity(emb1, emb2)
#         if sim < 0.7:
#             # print(f"Not similar ({sim:.2f}):")
#             # print(f"\t{df[k][idx[i]]}")
#             # print(f"\t{df['en'][idx[i]]}")
#             df[k][idx[i]] = "__NULL__"
#         sims.append(sim)
    
#     sims = np.array(sims)
#     print(f"Removal at 0.85: {sum(sims < 0.85) / emb.shape[0] * 100:.2f}% elements.")
#     print(f"Removal at 0.80: {sum(sims < 0.8) / emb.shape[0] * 100:.2f}% elements.")
#     print(f"Removal at 0.75: {sum(sims < 0.75) / emb.shape[0] * 100:.2f}% elements.")
#     print(f"Removal at 0.70: {sum(sims < 0.7) / emb.shape[0] * 100:.2f}% elements.")
#     print(f"Removal at 0.65: {sum(sims < 0.65) / emb.shape[0] * 100:.2f}% elements.")
#     print(f"Removal at 0.60: {sum(sims < 0.60) / emb.shape[0] * 100:.2f}% elements.")


dataset2 = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "dev": dev_dataset
})

dataset2["train"].to_json("./ted-multiling-filtered2/train.json")
dataset2["test"].to_json("./ted-multiling-filtered2/test.json")
dataset2["dev"].to_json("./ted-multiling-filtered2/dev.json")