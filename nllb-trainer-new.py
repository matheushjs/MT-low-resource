import transformers, datasets, torch
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys, os, re, time, unicodedata, pickle, html, copy, shutil
import sacrebleu
import wandb
from iso639 import Language
from pathlib import Path
from sacremoses import MosesPunctNormalizer
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    get_constant_schedule, get_constant_schedule_with_warmup,
    logging, EarlyStoppingCallback, DataCollatorForSeq2Seq
)
from transformers.optimization import Adafactor
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime as dt
from comet import download_model, load_from_checkpoint
from torch.utils.data import DataLoader, Dataset as torchDataset
from torch.optim import AdamW

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)


parser = argparse.ArgumentParser(prog='NLLB Model Trainer')
parser.add_argument("--lang-pairs",
        help="Comma-separated language pairs to use. Direction matters.",
        required=True,
        type=lambda x: x.split(","))
parser.add_argument("--training-steps",
        help="Number of training steps with various language pairs.",
        type=int,
        default=20000)
parser.add_argument("--post-training-steps",
        help="Number of training steps with only hy-en language pair.",
        type=int,
        default=20000)
parser.add_argument("--epochs",
        help="Number of epochs to train.",
        type=int,
        default=5)
parser.add_argument("--main-lang-pair",
        help="Main language pair. Direction matters.",
        default="en-hy")
parser.add_argument("--load-existing",
        help="Load specific checkpoint.",
        default="",
        type=str)
parser.add_argument("--skip-test",
        help="Should we test the model?",
        action='store_true')
parser.add_argument("--train-from-scratch",
        help="Load NLLB without loading the weights.",
        action='store_true')
parser.add_argument("--patience",
        help="Patience of early stopping.",
        type=int,
        default=50)
parser.add_argument("--post-patience",
        help="Patience of early stopping in post-training.",
        type=int,
        default=50)
parser.add_argument("--batch-size",
        help="Size of batches to process in the GPU. Increases GPU memory requirement.",
        type=int,
        default=16)
parser.add_argument("--tok-max-length",
        help="Truncate token sequences to this length. Increases GPU memory requirement.",
        type=int,
        default=256)
parser.add_argument("--limit-train-corpus",
        help="Truncates training corpus for each language to this number of sentences.",
        type=int,
        default=-1)
parser.add_argument("--limit-main-corpus",
        help="Truncates training corpus for the main language pair.",
        type=int,
        default=-1)
parser.add_argument("--limit-test-samples",
        help="Tests on only the first N samples.",
        type=int,
        default=-1)
parser.add_argument("--middle-limit-test-samples",
        help="Middle-tests on only the first N samples.",
        type=int,
        default=-1)
parser.add_argument("--gradient-accumulation-steps",
        help="Number of steps to accumulate the gradients before updating weights.",
        type=int,
        default=2)
parser.add_argument("--learning-rate",
        help="Learning rate for training.",
        type=float,
        default=5e-5)
parser.add_argument("--post-learning-rate",
        help="Learning rate for post-training. If -1, uses learning_rate / 10.",
        type=float,
        default=-1.0)
parser.add_argument("--weight-decay",
        help="Weight decay for training.",
        type=float,
        default=0.001)
parser.add_argument("--max-grad-norm",
        help="Gradient clipping for training.",
        type=float,
        default=0.5)
parser.add_argument("--warmup-ratio",
        help="Warmup ratio for training. Can be integer for no. of steps.",
        type=float,
        default=0.03)
parser.add_argument("--eval-steps",
        help="Frequency of evaluation.",
        type=int,
        default=100)
parser.add_argument("--post-eval-steps",
        help="Frequency of evaluation in post-training.",
        type=int,
        default=100)
parser.add_argument("--reset-prob",
        help="Probability of resetting a layer.",
        type=float,
        default=0)
parser.add_argument("--no-dropout",
        help="Should we remove dropout?",
        action='store_true')
parser.add_argument("--eval-all-langs",
        help="Should we evaluate on all languages during pre-training?",
        action='store_true')
#args = parser.parse_args(["--lang-pairs", "en-ko,en-hy"])
args = parser.parse_args()

