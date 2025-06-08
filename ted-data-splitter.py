import datasets
from datasets import load_dataset
import numpy as np
import pandas as pd
import sys, os, re

if __name__ == "__main__":
    dataset = load_dataset("json", data_files={
        "train": "./ted-multiling-filtered/train.json",
        "test": "./ted-multiling-filtered/test.json",
        "dev": "./ted-multiling-filtered/dev.json"
    })

    if 'pt' in dataset["train"].column_names:
        dataset = dataset.remove_columns("pt")
    dataset = dataset.rename_column("pt-br", "pt")

    colNames = dataset["train"].column_names

    for lang in colNames:
        print(lang)

        new_data = dataset.select_columns([lang]) \
                    .filter(lambda x: x[lang] != "__NULL__", num_proc=4)
        
        lines = new_data["train"][lang]
        lines.extend(new_data["dev"][lang])

        with open(f"data/{lang}.txt", "w+") as fp:
            fp.write("\n".join(lines))

    sys.exit()