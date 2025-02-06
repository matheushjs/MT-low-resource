import transformers, datasets, torch
import bitsandbytes as bnb
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys, os, re, time, unicodedata, pickle, html
import sacrebleu
import wandb
from iso639 import Language
from pathlib import Path
from sacremoses import MosesPunctNormalizer
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    get_constant_schedule_with_warmup, BitsAndBytesConfig, logging, EarlyStoppingCallback
)
from transformers.optimization import Adafactor
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig, setup_chat_format
from datetime import datetime as dt
from comet import download_model, load_from_checkpoint
from torch.utils.data import DataLoader, Dataset as torchDataset

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

