import transformers, datasets, torch
import bitsandbytes as bnb
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
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    get_constant_schedule_with_warmup, get_constant_schedule,
    BitsAndBytesConfig, logging, EarlyStoppingCallback, DefaultDataCollator
)
from transformers.optimization import Adafactor
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig, setup_chat_format
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


parser = argparse.ArgumentParser(prog='Llama Model Trainer')
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
        help="Load model without loading the weights.",
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
        help="Probability of resetting a layer (Not used).",
        type=float,
        default=0)
parser.add_argument("--lora-r",
        help="LoRA r parameter.",
        type=int,
        default=64)
parser.add_argument("--lora-alpha",
        help="LoRA alpha parameter.",
        type=int,
        default=128)
parser.add_argument("--lora-dropout",
        help="LoRA layers dropout probability.",
        type=float,
        default=0.05)
parser.add_argument("--eval-all-langs",
        help="Should we evaluate on all languages during pre-training?",
        action='store_true')
parser.add_argument("--do-complementary-test",
        help="Should we get metrics on the complementary test dataset?" ,
        action='store_true')

#args = parser.parse_args(["--lang-pairs", "en-ko,en-hy"])
args = parser.parse_args()

if args.main_lang_pair not in args.lang_pairs:
    print("Main language pair must be included in --lang-pairs.")
    sys.exit(1)

if args.post_training_steps > 0 and args.training_steps == 0:
    print("If --post-training-steps is non-zero, --training-steps must also be non-zero.")
    sys.exit(2)

engfirst = all([ i.split("-")[0] == 'en' for i in args.lang_pairs ])
englast  = all([ i.split("-")[1] == 'en' for i in args.lang_pairs ])
if not (engfirst ^ englast):
    print("In --lang-pairs, either 'en' always comes first, or it always comes last. The lang pairs provided are mixed.")
    sys.exit(3)

print("Beginning program.")
print("Command executed: {}".format(' '.join(sys.argv)))
for i, j in args._get_kwargs():
    print("{}: {}".format(i, j))
print("==================")

# Check language codes at: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
FLORES_TO_PART3 = {'bam_Latn':'bam','vec_Latn':'vec','prs_Arab':'prs','tuk_Latn':'tuk','tgl_Latn':'tgl','fij_Latn':'fij','nld_Latn':'nld','luo_Latn':'luo','arb_Latn':'arb','run_Latn':'run','eus_Latn':'eus','lim_Latn':'lim','kan_Knda':'kan','ydd_Hebr':'ydd','bel_Cyrl':'bel','kik_Latn':'kik','mlt_Latn':'mlt','kmb_Latn':'kmb','mar_Deva':'mar','min_Latn':'min','umb_Latn':'umb','kas_Arab':'kas','ewe_Latn':'ewe','lmo_Latn':'lmo','wol_Latn':'wol','tam_Taml':'tam','fao_Latn':'fao','slk_Latn':'slk','deu_Latn':'deu','mya_Mymr':'mya','awa_Deva':'awa','kor_Hang':'kor','apc_Arab':'apc','hne_Deva':'hne','mkd_Cyrl':'mkd','kat_Geor':'kat','rus_Cyrl':'rus','aeb_Arab':'aeb','bjn_Arab':'bjn','hat_Latn':'hat','hau_Latn':'hau','glg_Latn':'glg','swe_Latn':'swe','mni_Beng':'mni','ben_Beng':'ben','fuv_Latn':'fuv','pap_Latn':'pap','mag_Deva':'mag','vie_Latn':'vie','twi_Latn':'twi','ces_Latn':'ces','lvs_Latn':'lvs','shn_Mymr':'shn','est_Latn':'est','kac_Latn':'kac','kab_Latn':'kab','sag_Latn':'sag','swh_Latn':'swh','ary_Arab':'ary','acq_Arab':'acq','oci_Latn':'oci','dik_Latn':'dik','ars_Arab':'ars','mal_Mlym':'mal','gla_Latn':'gla','kea_Latn':'kea','hin_Deva':'hin','tpi_Latn':'tpi','srp_Cyrl':'srp','lit_Latn':'lit','crh_Latn':'crh','tha_Thai':'tha','azj_Latn':'azj','tur_Latn':'tur','bug_Latn':'bug','por_Latn':'por','azb_Arab':'azb','tir_Ethi':'tir','ukr_Cyrl':'ukr','tum_Latn':'tum','pol_Latn':'pol','epo_Latn':'epo','ace_Arab':'ace','nso_Latn':'nso','dan_Latn':'dan','tel_Telu':'tel','scn_Latn':'scn','lij_Latn':'lij','ltz_Latn':'ltz','ilo_Latn':'ilo','dzo_Tibt':'dzo','heb_Hebr':'heb','quy_Latn':'quy','bho_Deva':'bho','knc_Arab':'knc','pes_Arab':'pes','gle_Latn':'gle','szl_Latn':'szl','zho_Hans':'zho','kam_Latn':'kam','bjn_Latn':'bjn','lus_Latn':'lus','pbt_Arab':'pbt','spa_Latn':'spa','som_Latn':'som','hye_Armn':'hye','hun_Latn':'hun','lao_Laoo':'lao','dyu_Latn':'dyu','zul_Latn':'zul','nno_Latn':'nno','ron_Latn':'ron','san_Deva':'san','xho_Latn':'xho','hrv_Latn':'hrv','lug_Latn':'lug','bem_Latn':'bem','grn_Latn':'grn','khm_Khmr':'khm','taq_Latn':'taq','urd_Arab':'urd','slv_Latn':'slv','pag_Latn':'pag','zho_Hant':'zho','tzm_Tfng':'tzm','nob_Latn':'nob','gaz_Latn':'gaz','isl_Latn':'isl','arb_Arab':'arb','sun_Latn':'sun','mri_Latn':'mri','asm_Beng':'asm','yue_Hant':'yue','ita_Latn':'ita','ban_Latn':'ban','zsm_Latn':'zsm','bod_Tibt':'bod','ibo_Latn':'ibo','kaz_Cyrl':'kaz','jpn_Jpan':'jpn','smo_Latn':'smo','npi_Deva':'npi','ckb_Arab':'ckb','afr_Latn':'afr','mos_Latn':'mos','min_Arab':'min','lin_Latn':'lin','cym_Latn':'cym','sna_Latn':'sna','ory_Orya':'ory','kin_Latn':'kin','acm_Arab':'acm','kas_Deva':'kas','lua_Latn':'lua','kbp_Latn':'kbp','ace_Latn':'ace','ceb_Latn':'ceb','als_Latn':'als','fra_Latn':'fra','nus_Latn':'nus','guj_Gujr':'guj','snd_Arab':'snd','jav_Latn':'jav','tgk_Cyrl':'tgk','ell_Grek':'ell','bak_Cyrl':'bak','ayr_Latn':'ayr','uig_Arab':'uig','sin_Sinh':'sin','ast_Latn':'ast','srd_Latn':'srd','sot_Latn':'sot','fin_Latn':'fin','tat_Cyrl':'tat','knc_Latn':'knc','nya_Latn':'nya','ssw_Latn':'ssw','yor_Latn':'yor','pan_Guru':'pan','bul_Cyrl':'bul','kir_Cyrl':'kir','bos_Latn':'bos','aka_Latn':'aka','fur_Latn':'fur','ltg_Latn':'ltg','amh_Ethi':'amh','kon_Latn':'kon','kmr_Latn':'kmr','uzn_Latn':'uzn','ind_Latn':'ind','mai_Deva':'mai','eng_Latn':'eng','plt_Latn':'plt','arz_Arab':'arz','taq_Tfng':'taq','tsn_Latn':'tsn','war_Latn':'war','tso_Latn':'tso','ajp_Arab':'ajp','khk_Cyrl':'khk','fon_Latn':'fon','cjk_Latn':'cjk','sat_Olck':'sat','cat_Latn':'cat'}
FLORES_TO_PART1 = {'bam_Latn':'bm','vec_Latn':None,'prs_Arab':None,'tuk_Latn':'tk','tgl_Latn':'tl','fij_Latn':'fj','nld_Latn':'nl','luo_Latn':None,'arb_Latn':None,'run_Latn':'rn','eus_Latn':'eu','lim_Latn':'li','kan_Knda':'kn','ydd_Hebr':None,'bel_Cyrl':'be','kik_Latn':'ki','mlt_Latn':'mt','kmb_Latn':None,'mar_Deva':'mr','min_Latn':None,'umb_Latn':None,'kas_Arab':'ks','ewe_Latn':'ee','lmo_Latn':None,'wol_Latn':'wo','tam_Taml':'ta','fao_Latn':'fo','slk_Latn':'sk','deu_Latn':'de','mya_Mymr':'my','awa_Deva':None,'kor_Hang':'ko','apc_Arab':None,'hne_Deva':None,'mkd_Cyrl':'mk','kat_Geor':'ka','rus_Cyrl':'ru','aeb_Arab':None,'bjn_Arab':None,'hat_Latn':'ht','hau_Latn':'ha','glg_Latn':'gl','swe_Latn':'sv','mni_Beng':None,'ben_Beng':'bn','fuv_Latn':None,'pap_Latn':None,'mag_Deva':None,'vie_Latn':'vi','twi_Latn':'tw','ces_Latn':'cs','lvs_Latn':None,'shn_Mymr':None,'est_Latn':'et','kac_Latn':None,'kab_Latn':None,'sag_Latn':'sg','swh_Latn':None,'ary_Arab':None,'acq_Arab':None,'oci_Latn':'oc','dik_Latn':None,'ars_Arab':None,'mal_Mlym':'ml','gla_Latn':'gd','kea_Latn':None,'hin_Deva':'hi','tpi_Latn':None,'srp_Cyrl':'sr','lit_Latn':'lt','crh_Latn':None,'tha_Thai':'th','azj_Latn':'az','tur_Latn':'tr','bug_Latn':None,'por_Latn':'pt','azb_Arab':None,'tir_Ethi':'ti','ukr_Cyrl':'uk','tum_Latn':None,'pol_Latn':'pl','epo_Latn':'eo','ace_Arab':None,'nso_Latn':None,'dan_Latn':'da','tel_Telu':'te','scn_Latn':None,'lij_Latn':None,'ltz_Latn':'lb','ilo_Latn':None,'dzo_Tibt':'dz','heb_Hebr':'he','quy_Latn':None,'bho_Deva':None,'knc_Arab':None,'pes_Arab':None,'gle_Latn':'ga','szl_Latn':None,'zho_Hans':'zh','kam_Latn':None,'bjn_Latn':None,'lus_Latn':None,'pbt_Arab':None,'spa_Latn':'es','som_Latn':'so','hye_Armn':'hy','hun_Latn':'hu','lao_Laoo':'lo','dyu_Latn':None,'zul_Latn':'zu','nno_Latn':'nn','ron_Latn':'ro','san_Deva':'sa','xho_Latn':'xh','hrv_Latn':'hr','lug_Latn':'lg','bem_Latn':None,'grn_Latn':'gn','khm_Khmr':'km','taq_Latn':None,'urd_Arab':'ur','slv_Latn':'sl','pag_Latn':None,'zho_Hant':'zh','tzm_Tfng':None,'nob_Latn':'nb','gaz_Latn':None,'isl_Latn':'is','arb_Arab':'ar','sun_Latn':'su','mri_Latn':'mi','asm_Beng':'as','yue_Hant':None,'ita_Latn':'it','ban_Latn':None,'zsm_Latn':None,'bod_Tibt':'bo','ibo_Latn':'ig','kaz_Cyrl':'kk','jpn_Jpan':'ja','smo_Latn':'sm','npi_Deva':None,'ckb_Arab':None,'afr_Latn':'af','mos_Latn':None,'min_Arab':None,'lin_Latn':'ln','cym_Latn':'cy','sna_Latn':'sn','ory_Orya':None,'kin_Latn':'rw','acm_Arab':None,'kas_Deva':'ks','lua_Latn':None,'kbp_Latn':None,'ace_Latn':None,'ceb_Latn':None,'als_Latn':None,'fra_Latn':'fr','nus_Latn':None,'guj_Gujr':'gu','snd_Arab':'sd','jav_Latn':'jv','tgk_Cyrl':'tg','ell_Grek':'el','bak_Cyrl':'ba','ayr_Latn':None,'uig_Arab':'ug','sin_Sinh':'si','ast_Latn':None,'srd_Latn':'sc','sot_Latn':'st','fin_Latn':'fi','tat_Cyrl':'tt','knc_Latn':None,'nya_Latn':'ny','ssw_Latn':'ss','yor_Latn':'yo','pan_Guru':'pa','bul_Cyrl':'bg','kir_Cyrl':'ky','bos_Latn':'bs','aka_Latn':'ak','fur_Latn':None,'ltg_Latn':None,'amh_Ethi':'am','kon_Latn':'kg','kmr_Latn':None,'uzn_Latn':None,'ind_Latn':'id','mai_Deva':None,'eng_Latn':'en','plt_Latn':None,'arz_Arab':None,'taq_Tfng':None,'tsn_Latn':'tn','war_Latn':None,'tso_Latn':'ts','ajp_Arab':None,'khk_Cyrl':None,'fon_Latn':None,'cjk_Latn':None,'sat_Olck':None,'cat_Latn':'ca'}
PART3_TO_FLORES = { j: i for i,j in FLORES_TO_PART3.items() }
PART1_TO_FLORES = { j: i for i,j in FLORES_TO_PART1.items() }

EXPERIMENT_NAME = "-".join([
    'results-llama',
    '-'.join(sorted(args.lang_pairs)),
    f'mainpair{args.main_lang_pair}',
    f'stepsA{args.training_steps}',
    f'stepsB{args.post_training_steps}',
    f'batchsize{args.batch_size}'
])
MODEL_SAVE_PATH = './checkpoints/{}'.format(EXPERIMENT_NAME)
POST_MODEL_SAVE_PATH = './checkpoints/posttrain-{}'.format(EXPERIMENT_NAME)

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def dataset_for_llama(tokenizer, lang_pairs=args.lang_pairs):
    """Generate Dataset for Llama or Qwen"""
    
    dataset = load_dataset("json", data_files={
        "train": "./ted-multiling-filtered/train.json",
        "test": "./ted-multiling-filtered/test.json",
        "dev": "./ted-multiling-filtered/dev.json"
    })

    if 'pt' in dataset["train"].column_names:
        dataset = dataset.remove_columns("pt")
    dataset = dataset.rename_column("pt-br", "pt")

    paired_sentences_datasets = []
    main_lang_eval_size = -1

    for lang_pair in lang_pairs:
        lang1, lang2 = lang_pair.split("-")
        name1, name2 = Language.from_part1(lang1).name, Language.from_part1(lang2).name

        new_data = dataset.select_columns([lang1, lang2]) \
                    .filter(lambda x: x[lang1] != "__NULL__" and x[lang2] != "__NULL__", num_proc=4) \
                    .map(lambda x: {
                        'lang1': lang1,
                        'lang2': lang2,
                        'name1': name1,
                        'name2': name2,
                        'sentence1': x[lang1],
                        'sentence2': x[lang2]
                    }, num_proc=4) \
                    .remove_columns([lang1, lang2])

        if lang_pair == args.main_lang_pair and args.limit_main_corpus > 0:
            print(f"Reducing main language train corpus from {len(new_data["train"])} to {args.limit_main_corpus}.")
            new_data["train"] = new_data["train"].shuffle().select(range(args.limit_main_corpus))

        print(f"Length of {lang_pair} dataset:", len(new_data["train"]), len(new_data["dev"]), len(new_data["test"]))

        # TODO: implement this properly (with a custom Dataset subclass)
        # For now, we truncate OR expand all corpuses to the limit_train_corpus
        if args.limit_train_corpus > 0:
            if len(new_data["train"]) > args.limit_train_corpus:
                new_data["train"] = new_data["train"].shuffle().select(range(args.limit_train_corpus))
            elif len(new_data["train"]) < args.limit_train_corpus:
                multiplier = int(np.ceil(args.limit_train_corpus / len(new_data["train"])))
                new_data["train"] = datasets.concatenate_datasets([ new_data["train"] for k in range(multiplier) ])
                new_data["train"] = new_data["train"].select(range(args.limit_train_corpus))

        if lang_pair == args.main_lang_pair:
            main_lang_eval_size = len(new_data["dev"])

        paired_sentences_datasets.append(new_data)

    if args.eval_all_langs:
        if main_lang_eval_size <= 0: raise Exception("Something wrong with main_lang_eval_size.")
        for d in paired_sentences_datasets:
            d["dev"] = d["dev"].select(range(main_lang_eval_size))

    lengths = [ len(i["train"]) for i in paired_sentences_datasets ]
    multipliers = [ max(lengths) // i for i in lengths ]
    leftovers = [ max(lengths) % i for i in lengths ]
    to_concatenate = []

    print("Multipliers for each dataset: ", list(zip(lang_pairs, multipliers)))

    for idx, (a, b) in enumerate(zip(multipliers, leftovers)):
        to_concatenate.extend([ paired_sentences_datasets[idx]["train"] ] * a)
        
        if b > 0:
            t = to_concatenate[-1]
            new_samples = t.shuffle().select(range(b))
            to_concatenate[-1] = datasets.concatenate_datasets([t, new_samples])
    
    new_dataset = DatasetDict({
        "train": datasets.concatenate_datasets(to_concatenate),
        "test": datasets.concatenate_datasets([ i["test"] for i in paired_sentences_datasets ]),
        "dev": datasets.concatenate_datasets([ i["dev"] for i in paired_sentences_datasets ])
    })

    return new_dataset.shuffle()

def translate(
    input_ids, attention_mask, tokenizer, model,
    a=32, b=3, max_input_length=256, num_beams=3, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""

    model.eval()
    result = model.generate(
        input_ids=torch.Tensor(input_ids).to(model.device),
        attention_mask=torch.Tensor(attention_mask).to(model.device),
        num_beams=num_beams,
        max_new_tokens=int(a + b * len(input_ids)),
        do_sample=False, top_p=None, temperature=None,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(result[0], skip_special_tokens=True)
    return "".join(text.split("assistant")[1:]).strip()

def get_translations(dataset, tokenizer, model, limit_samples=-1, do_print=True):
    translations = []
    try:
        for idx, row in enumerate(dataset):
            lang1 = row["lang1"]
            lang2 = row["lang2"]
            X_eng = row["sentence1"]
            X_hye = row["sentence2"]
            input_ids = row["input_ids"]
            attention_mask = row["attention_mask"]
            eng_to_hye = translate(input_ids, attention_mask, tokenizer, model)

            translations.append((X_hye, X_eng, eng_to_hye))

            if do_print and idx < 20:
                print(f"{lang2} (target): ", X_hye)
                print(f"{lang1} (source): ", X_eng)
                print("Translated: ", eng_to_hye)
                print("=============================")

            if limit_samples > 0 and idx >= (limit_samples - 1):
                print("Interrupting get_translations() due to 'limit_samples' argument.")
                break

    except KeyboardInterrupt:
        print("Caught Ctrl+C or SIGINT. Interrupting testing.")

    return translations

def find_all_linear_names(model):
    """For use with LoRA, which trains only linear layers."""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_scores(translations, do_comet=False):
    translations = [ i for i in translations ]
    bleu_score  = None
    chrf_score  = None
    comet_score = None

    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

    # We should filter empty sentences, or the package will give an error.
    idxs = []
    for idx, row in enumerate(translations):
        if "" in row or " " in row:
            idxs.append(idx)
    # ALWAYS remove in inverted order :)
    for i in idxs[::-1]:
        print(f"Removing row {i}, with contents {translations[i]}")
        translations.pop(i) # del translations[i]

    try:
        num_removed = len(idxs)
        total = num_removed + len(translations)

        print(f"Percentage of samples removed: {num_removed / total * 100:.2f}%")

        bleu_score = bleu_calc.corpus_score([i[0] for i in translations], [[i[2] for i in translations]])
        bleu_score.score = bleu_score.score * (len(translations) / total)

        chrf_score = chrf_calc.corpus_score([i[0] for i in translations], [[i[2] for i in translations]])
        chrf_score.score = chrf_score.score * (len(translations) / total)

        if do_comet:
            data = []
            for xyz in translations:
                data.append({
                    "src": xyz[1],
                    "mt": xyz[2],
                    "ref": xyz[0]
                })

            try:
                model_path = download_model("Unbabel/wmt22-comet-da", local_files_only=True)
            except:
                model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)

            model_output = comet_model.predict(data, gpus=1, num_workers=0, progress_bar=False)
            comet_score = model_output.system_score

    except Exception as e:
        print("Failed to calculate BLEU, ChrF or Comet scores.")
        print(e)

    return bleu_score, chrf_score, comet_score

if __name__ == "__main__":
    timer_global = dt.today()

    if args.load_existing != "":
        checkpoint = args.load_existing
    else:
        checkpoint = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint) #, padding_size="left")

    # Set torch dtype and attention implementation
    # NOTE: This might cause result differences between Euler and KIT clusters
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    print(f"Compute capability: {torch.cuda.get_device_capability()[0]}. Loading model with torch_dtype {torch_dtype}.")

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    if args.training_steps > 0:
        print("Loading QLoRA quantized model.")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=bnb_config
        )
    else:
        print("Loading model in float16/bfloat16 precision.")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype
        )
        model.to("cuda")

    # tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    #tokenizer.chat_template = None
    #model, tokenizer = setup_chat_format(model, tokenizer)

    print("===== CHAT TEMPLATE ====")
    print("Padding side:", tokenizer.padding_side)
    print(tokenizer.chat_template)
    print("========================")

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("You might want to consider resizing the model embeddings?")
        #model.resize_token_embeddings(len(tokenizer))

    main_lang1 = args.main_lang_pair.split("-")[0]
    main_lang2 = args.main_lang_pair.split("-")[1]

    batch_size = args.batch_size  # 32 already doesn't fit well to 15GB of GPU memory
    max_length = args.tok_max_length  # token sequences will be truncated
    training_steps = args.training_steps
    post_training_steps = args.post_training_steps
    patience = args.patience
    losses = []
    post_losses = []
    devlosses = []
    middle_translations = []
    translations = []
    dev_translations = []
    training_step_counter = 0
    post_training_step_counter = 0
    time_begin = time.time()

    timer_dataset_preparation = dt.today()
    datasets = dataset_for_llama(tokenizer, lang_pairs=args.lang_pairs)

    # We have a potential problem of concurrent usage of the tokenizer, so let's just
    # instantiate multiple of them
    tokenizers = dict()
    for lang_pair in args.lang_pairs:
        lang1, lang2 = lang_pair.split("-")
        newTok = copy.deepcopy(tokenizer)
        newTok.src_lang = PART1_TO_FLORES[lang1]
        newTok.tgt_lang = PART1_TO_FLORES[lang2]

        if lang1 not in tokenizers:
            tokenizers[lang1] = {lang2: newTok}
        else:
            tokenizers[lang1][lang2] = newTok

    def _tokenize_train_fn(row):
        instruction = f"You are a professional translator proficient in translating {row["name1"]} text to {row["name2"]}. " + \
                       "Your responses should contain only the translated text without any additional commentary."
        prompt = f"Translate this from {row["name1"]} to {row["name2"]}: {row["sentence1"]}"
        row_json = [{"role": "system", "content": instruction },
                    {"role": "user", "content": prompt },
                    {"role": "assistant", "content": row["sentence2"] } ]
        myTokenizer = tokenizers[row["lang1"]][row["lang2"]]
        text = myTokenizer.apply_chat_template(row_json, tokenize=False)
        row["text"] = text
        return row

    def _tokenize_test_fn(row):
        instruction = f"You are a professional translator proficient in translating {row["name1"]} text to {row["name2"]}. " + \
                       "Your responses should contain only the translated text without any additional commentary."
        prompt = f"Translate this from {row["name1"]} to {row["name2"]}: {row["sentence1"]}"
        row_json = [{"role": "system", "content": instruction },
                    {"role": "user", "content": prompt }]
        myTokenizer = tokenizers[row["lang1"]][row["lang2"]]
        text = myTokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=True)
        inputs = myTokenizer(text, return_tensors="pt", padding=True, max_length=args.tok_max_length, truncation=True)
        row["prompt"] = text
        row["input_ids"] = inputs["input_ids"]
        row["attention_mask"] = inputs["attention_mask"]
        return row

    train_dataset      = datasets["train"] \
                            .map(_tokenize_train_fn, num_proc=4)
    post_train_dataset = train_dataset \
                            .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2) \
                            .shuffle()

    if args.eval_all_langs:
        full_dev_dataset = datasets["dev"].map(_tokenize_train_fn, num_proc=4)
        dev_dataset  = full_dev_dataset
        post_dev_dataset  = full_dev_dataset \
                    .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
    else:
        full_dev_dataset = datasets["dev"].map(_tokenize_train_fn, num_proc=4)
        dev_dataset = full_dev_dataset \
                    .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
        post_dev_dataset = dev_dataset
    
    full_test_dataset = datasets["test"].map(_tokenize_test_fn, num_proc=4)

    test_dataset = full_test_dataset \
                    .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
    
    if len(args.lang_pairs) > 1:
        if main_lang1 == "en":
            complement_test_dataset = full_test_dataset \
                            .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] != main_lang2)
        else:
            complement_test_dataset = full_test_dataset \
                            .filter(lambda x: x["lang1"] != main_lang1 and x["lang2"] == main_lang2)
    else:
        complement_test_dataset = None

    print("Printing some samples of the dataset.")
    print("lang1: ", "|".join(train_dataset["lang1"][:30]))
    print("lang2: ", "|".join(train_dataset["lang2"][:30]))
    print("name1: ", "|".join(train_dataset["name1"][:30]))
    print("name2: ", "|".join(train_dataset["name2"][:30]))
    print("sentence1:\n", "\n".join(train_dataset["sentence1"][:5]))
    print("sentence2:\n", "\n".join(train_dataset["sentence2"][:5]))
    print("text fields:")
    for j in range(5):
        print(train_dataset[j]["text"])

    print("Printing some samples of the Evaluation dataset.")
    print("lang1: ", "|".join(dev_dataset["lang1"][:30]))
    print("lang2: ", "|".join(dev_dataset["lang2"][:30]))
    print("sentence1:\n", "\n".join(dev_dataset["sentence1"][:5]))
    print("sentence2:\n", "\n".join(dev_dataset["sentence2"][:5]))
    print("text fields:")
    for j in range(5):
        print(train_dataset[j]["text"])

    print(f"Dataset sizes:\ntrain: {len(train_dataset)}\npost_train: {len(post_train_dataset)}\ndev: {len(dev_dataset)}\npost_dev: {len(post_dev_dataset)}\ntest: {len(test_dataset)}")

    print("Timer (dataset preparation):", dt.today() - timer_dataset_preparation)

    if args.post_learning_rate < 0:
        post_lr = args.learning_rate / 10
    else:
        post_lr = args.post_learning_rate

    cleanup()
    if args.training_steps > 0:
        modules = find_all_linear_names(model)

        # LoRA config
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules
        )
        # try:
        #     model, tokenizer = setup_chat_format(model, tokenizer)
        # except Exception as e:
        #     print(e)
        model = get_peft_model(model, peft_config)

        run = wandb.init(
            entity="elfmathews-university-of-tsukuba",
            project=EXPERIMENT_NAME, 
            job_type="training", 
            anonymous="allow"
        )

        epoch_batch_count = len(train_dataset) / args.batch_size
        warmup_steps = int(args.warmup_ratio * epoch_batch_count)

        #Hyperparamter
        training_arguments = SFTConfig(
            output_dir=MODEL_SAVE_PATH,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            num_train_epochs=args.epochs,
            max_steps=args.training_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            eval_accumulation_steps=50,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_steps=1,
            logging_strategy="steps",
            learning_rate=args.learning_rate if len(args.lang_pairs) > 1 else post_lr,
            fp16=False,
            bf16=False,
            weight_decay=args.weight_decay,
            group_by_length=False,
            max_seq_length=args.tok_max_length,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=warmup_steps,
            lr_scheduler_type="constant_with_warmup",
            dataloader_num_workers=4,
            dataset_text_field="text", # This argument was on Trainer
            #max_length=512, # Idk what's going on. This is the latest version of TRL but it doesn't seem like it.
            report_to="wandb",
            save_safetensors=False,
            eval_on_start=True
        )

        # Setting sft parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            peft_config=peft_config,
            #max_seq_length=512,
            #dataset_text_field="text",
            processing_class=tokenizer,
            args=training_arguments,
            #optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(args.patience)]
        )

        model.config.use_cache = False
        timer_pretraining = dt.today()
        try:
            train_result = trainer.train()
        except KeyboardInterrupt:
            print("Caught Ctrl+C or SIGINT. Interrupting pre-training and proceeding to middle testing.")
        print("Timer (pre-training):", dt.today() - timer_pretraining)

        # Cleanup checkpoints
        print("Cleaning up checkpoints.")
        for filename in os.listdir(MODEL_SAVE_PATH + "/"):
            if filename.startswith("checkpoint-"):
                dirname = os.path.join(MODEL_SAVE_PATH, filename)
                print(f"Removing: {dirname}")
                shutil.rmtree(dirname)

        wandb.finish()

        for i in trainer.state.log_history:
            if 'loss' in i:
                losses.append(i["loss"])
        # max_train_samples = len(dataset["train"])
        # metrics["train_samples"] = len(dataset["train"])
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)

    if not args.skip_test and args.post_training_steps > 0:
        model.config.use_cache = True
        timer_middle_testing = dt.today()

        print("Beginning middle testing.")
        middle_translations = get_translations(test_dataset, tokenizer, model, args.middle_limit_test_samples)

        print("\nStarting to score the test dataset (middle testing).")
        print(f"Number of sentences: {len(middle_translations)}")
        test_scores = get_scores(middle_translations, do_comet=False)
        print(test_scores[0])
        print(test_scores[1])
        print("Timer (middle testing):", dt.today() - timer_middle_testing)

    time_after_train = time.time()

    if args.post_training_steps > 0:
        run = wandb.init(
            entity="elfmathews-university-of-tsukuba",
            project="posttrain-" + EXPERIMENT_NAME, 
            job_type="training", 
            anonymous="allow"
        )

        epoch_batch_count = len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps)
        warmup_steps = int(args.warmup_ratio * epoch_batch_count)

        #Hyperparamter
        training_arguments = SFTConfig(
            output_dir=POST_MODEL_SAVE_PATH,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            num_train_epochs=args.epochs,
            max_steps=args.post_training_steps,
            eval_strategy="steps",
            eval_steps=args.post_eval_steps,
            eval_accumulation_steps=50,
            save_strategy="steps",
            save_steps=args.post_eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_steps=1,
            logging_strategy="steps",
            learning_rate=post_lr,
            fp16=False,
            bf16=False,
            weight_decay=args.weight_decay,
            group_by_length=False,
            max_seq_length=args.tok_max_length,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=warmup_steps,
            lr_scheduler_type="constant_with_warmup",
            dataloader_num_workers=4,
            dataset_text_field="text", # This argument was on Trainer
            #max_length=512, # Idk what's going on. This is the latest version of TRL but it doesn't seem like it.
            report_to="wandb",
            save_safetensors=False
        )

        # Setting sft parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=post_train_dataset,
            eval_dataset=post_dev_dataset,
            peft_config=peft_config,
            #max_seq_length=512,
            #dataset_text_field="text",
            processing_class=tokenizer,
            args=training_arguments,
            #optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(args.post_patience)]
        )

        model.config.use_cache = False
        timer_post_training = dt.today()
        try:
            post_train_result = trainer.train()
        except KeyboardInterrupt:
            print("Caught Ctrl+C or SIGINT. Interrupting post-training and proceeding to scoring.")
        print("Timer (post-training):", dt.today() - timer_post_training)

        # Cleanup checkpoints
        print("Cleaning up checkpoints.")
        for filename in os.listdir(POST_MODEL_SAVE_PATH + "/"):
            if filename.startswith("checkpoint-"):
                dirname = os.path.join(POST_MODEL_SAVE_PATH, filename)
                print(f"Removing: {dirname}")
                shutil.rmtree(dirname)

        wandb.finish()

        # Available: {'loss': 1.8642, 'grad_norm': 1.6934727430343628, 'learning_rate': 2e-05, 'mean_token_accuracy': 0.5831297039985657, 'epoch': 0.0022222222222222222, 'step': 1}
        for i in trainer.state.log_history:
            if 'loss' in i:
                post_losses.append(i["loss"])

    time_after_post_train = time.time()
    time_after_test = -1 # Initialize here so it exists even if we skip_test
    if not args.skip_test:
        try:
            model.dequantize()
        except ValueError:
            # The model is not quantized, but that's fine for testing.
            pass

        model.config.use_cache = True
        timer_post_testing = dt.today()

        print("Beginning testing on the test dataset (LRL only).")
        translations = get_translations(test_dataset, tokenizer, model, args.limit_test_samples)
        
        if complement_test_dataset != None:
            print("Beginning testing on the complementary test dataset (all but the LRL).")
            comp_translations = get_translations(complement_test_dataset, tokenizer, model, args.limit_test_samples)

        time_after_test = time.time()

        # Since memory is a problem for us, we delete the MT model before scoring with Comet.
        del model
        cleanup()
        
        all_bleu  = []
        all_chrf  = []
        all_comet = []

        print("\nStarting to score the test dataset (LRL only).")
        print(f"Number of sentences: {len(translations)}")
        test_scores = get_scores(translations, do_comet=True)
        print(test_scores[0])
        print(test_scores[1])
        print("COMET system score:", test_scores[2])
        all_bleu  += [test_scores[0].score] * len(translations)
        all_chrf  += [test_scores[1].score] * len(translations)
        all_comet += [test_scores[2]] * len(translations)
        print("Timer (post testing):", dt.today() - timer_post_testing)

        if args.do_complementary_test and complement_test_dataset != None:
            timer_comp_testing = dt.today()
            print("\nStarting to score the complementary test dataset (all but the LRL).")
            print(f"Number of sentences: {len(comp_translations)}")
            comp_test_scores = get_scores(comp_translations, do_comet=True)
            print("(comp_test)", comp_test_scores[0])
            print("(comp_test)", comp_test_scores[1])
            print("COMET system score:", comp_test_scores[2])
            all_bleu  += [comp_test_scores[0].score] * len(comp_translations)
            all_chrf  += [comp_test_scores[1].score] * len(comp_translations)
            all_comet += [comp_test_scores[2]] * len(comp_translations)
            print("Full test dataset average BLEU:", np.mean(all_bleu))
            print("Full test dataset average chrF2++:", np.mean(all_chrf))
            print("Full test dataset average COMET:", np.mean(all_comet))
            print("Timer (complementary testing):", dt.today() - timer_comp_testing)

    print("Timer (full program execution):", dt.today() - timer_global)

    # Do not use args.training_steps or args.post_training_steps
    if training_steps > 0 or post_training_steps > 0:
        try:
            print("Training epoch statistics:")
            for lang_pair in args.lang_pairs:
                lang1, lang2 = lang_pair.split("-")
                epochs_finished = train_df[lang_pair]["epoch_count"]
                remaining_percentage = len(train_df[lang_pair]["epoch_idx"]) / len(train_df[lang_pair][lang1])
                print(f"\t{lang_pair} - {epochs_finished} Epochs finished, and {remaining_percentage} remaining.")
        except:
            print("Failed to print epoch statistics, probably because model != nllb.")

    if len(losses) > 0:
        plt.figure()
        plt.scatter(np.arange(len(losses)), losses)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig("loss-{}.png".format(EXPERIMENT_NAME))

    if len(post_losses) > 0:
        plt.figure()
        plt.scatter(np.arange(len(post_losses)), post_losses)
        plt.xlabel("Post-training step")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig("postloss-{}.png".format(EXPERIMENT_NAME))

    today = dt.today()
    homedir = os.environ['HOME']
    resultsdir = "results-2025-{}-{}/".format(today.month, today.day)
    resultsfile = "results-{}.pickle".format(EXPERIMENT_NAME)

    Path(os.path.join(homedir, resultsdir)).mkdir(parents=True, exist_ok=True)

    time_end = time.time()
    elapsed = time_end - time_begin

    # This is dirty, but I consider results to be way too important
    #   to be lost due to non-existing variables
    results = {}
    try: results["losses"] = losses
    except: pass

    try: results["post_losses"] = post_losses
    except: pass

    try: results["translations"] = translations
    except: pass

    try: results["training_step_counter"] = training_step_counter
    except: pass

    try: results["post_training_step_counter"] = post_training_step_counter
    except: pass

    try: results["elapsed_time"] = elapsed
    except: pass

    try: results["time_to_train"] = time_after_train - time_begin
    except: pass

    try: results["time_to_post_train"] = time_after_post_train - time_after_train
    except: pass

    try: results["time_to_test"] = time_after_test - time_after_post_train
    except: pass

    try:
        pkldump(results, os.path.join(homedir, resultsdir, resultsfile))
    except:
        file = f"./{EXPERIMENT_NAME}.out"
        print(f"Something went wrong in saving results. Falling back to saving as string to {file}.\nYou should rename this file as soon as you can.")
        
        with open(file, "w") as fp:
            fp.write(str(results))