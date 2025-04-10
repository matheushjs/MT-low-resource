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
    'results-nllb',
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

def dataset_for_nllb(tokenizer, lang_pairs=args.lang_pairs):
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

        # TODO: Fix this dirty plan and implement a problem Dataset class
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
    text, tokenizer, model, src_lang, tgt_lang, 
    a=32, b=3, max_input_length=256, num_beams=3, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""

    tokenizer.src_lang = PART1_TO_FLORES[src_lang]
    tokenizer.tgt_lang = PART1_TO_FLORES[tgt_lang]
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, 
        max_length=max_input_length
    )

    model.eval() # turn off training mode
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(PART1_TO_FLORES[tgt_lang]),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

def reset_non_embedding_weights(model):
    random.seed(42)
    for name, module in model.named_modules():
        if any(x in name for x in ["embed", "shared", "lm_head"]):
            #print(f"Skipping embedding-related layer: {name}")
            continue
        if hasattr(module, "reset_parameters"):
            #print(f"Resetting: {name}")
            if random.random() < args.reset_prob:
                module.reset_parameters()
        else:
            #print(f"Skipping unresettable layer: {name}")
            pass

def get_translations(dataset, tokenizer, model, limit_samples=-1, do_print=True):
    translations = []
    try:
        for idx, row in enumerate(dataset):
            lang1 = row["lang1"]
            lang2 = row["lang2"]
            X_eng = row["sentence1"]
            X_hye = row["sentence2"]
            name1 = row["name1"]
            name2 = row["name2"]
            eng_to_hye = translate(X_eng, tokenizer, model, lang1, lang2)

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
    if args.load_existing != "":
        checkpoint = args.load_existing
    else:
        checkpoint = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint) #, padding_size="left")

    # Set torch dtype and attention implementation
    # NOTE: This might cause result differences between Euler and KIT clusters
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    print(f"Compute capability: {torch.cuda.get_device_capability()[0]}. Loading model with torch_dtype {torch_dtype}.")

    # Load model
    if args.training_steps > 0:
        if args.train_from_scratch and args.reset_prob > 0:
            print("Loading model for training with some layers reset.")
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            reset_non_embedding_weights(model)
        elif args.train_from_scratch and args.reset_prob <= 0:
            print("Loading model for training from scratch.")
            config = AutoConfig.from_pretrained(checkpoint)
            model  = AutoModelForSeq2SeqLM.from_config(config)
        else:
            print("Loading model for full regular training.")
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        if args.no_dropout:
            model.config.activation_dropout = 0
            model.config.attention_dropout = 0
            model.config.dropout = 0
            model.config.classifier_dropout = 0

        model.to("cuda")
    else:
        print("Loading model in float16/bfloat16 precision.")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype
        )
        model.to("cuda")

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
    losses = []  # with this list, I do very simple tracking of average loss
    post_losses = []
    devlosses = []
    middle_translations = []
    translations = []
    dev_translations = []
    training_step_counter = 0
    post_training_step_counter = 0
    time_begin = time.time()

    datasets = dataset_for_nllb(tokenizer, lang_pairs=args.lang_pairs)

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

    def _tokenize_fn(row):
        myTokenizer = tokenizers[row["lang1"]][row["lang2"]]
        model_inputs = myTokenizer(
            row["sentence1"],
            text_target=row["sentence2"],
            truncation=True,
            padding=False,
            max_length=args.tok_max_length,
        )
        for k in row.keys():
            model_inputs[k] = row[k]
        return model_inputs

    train_dataset      = datasets["train"] \
                            .map(_tokenize_fn, num_proc=4)
    post_train_dataset = train_dataset \
                            .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2) \
                            .shuffle()

    if args.eval_all_langs:
        full_dev_dataset = datasets["dev"].map(_tokenize_fn, num_proc=4)
        dev_dataset  = full_dev_dataset
        post_dev_dataset  = full_dev_dataset \
                    .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
    else:
        full_dev_dataset = datasets["dev"].map(_tokenize_fn, num_proc=4)
        dev_dataset = full_dev_dataset \
                    .filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
        post_dev_dataset = dev_dataset
    
    full_test_dataset = datasets["test"].map(_tokenize_fn, num_proc=4)

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
    print("input_ids:")
    for j in range(5):
        print(train_dataset[j]["input_ids"])
    print("labels:")
    for j in range(5):
        print(train_dataset[j]["labels"])

    print("Printing some samples of the Evaluation dataset.")
    print("lang1: ", "|".join(dev_dataset["lang1"][:30]))
    print("lang2: ", "|".join(dev_dataset["lang2"][:30]))
    print("sentence1:\n", "\n".join(dev_dataset["sentence1"][:5]))
    print("sentence2:\n", "\n".join(dev_dataset["sentence2"][:5]))
    print("input_ids:")
    for j in range(5):
        print(dev_dataset[j]["input_ids"])
    print("labels:")
    for j in range(5):
        print(dev_dataset[j]["labels"])

    print(f"Dataset sizes:\ntrain: {len(train_dataset)}\npost_train: {len(post_train_dataset)}\ndev: {len(dev_dataset)}\npost_dev: {len(post_dev_dataset)}\ntest: {len(test_dataset)}")

    cleanup()
    if args.training_steps > 0:
        run = wandb.init(
            entity="elfmathews-university-of-tsukuba",
            project=EXPERIMENT_NAME, 
            job_type="training", 
            anonymous="allow"
        )

        # optimizer = AdamW(
        #     [p for p in model.parameters() if p.requires_grad],
        #     lr=args.learning_rate,
        #     weight_decay=args.weight_decay
        # )
        # scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_ratio)

        # Data collator for seq2seq
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        epoch_batch_count = len(train_dataset) / args.batch_size
        warmup_steps = int(args.warmup_ratio * epoch_batch_count)

        #Hyperparamter
        training_arguments = Seq2SeqTrainingArguments(
            output_dir=MODEL_SAVE_PATH,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim="adamw_torch",
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
            learning_rate=args.learning_rate,
            fp16=True,
            bf16=False,
            weight_decay=args.weight_decay,
            group_by_length=False,
            #max_seq_length=args.tok_max_length,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=warmup_steps,
            lr_scheduler_type="constant_with_warmup",
            dataloader_num_workers=4,
            #max_length=512, # Idk what's going on. This is the latest version of TRL but it doesn't seem like it.
            report_to="wandb",
            save_safetensors=False,
            eval_on_start=True
        )

        # Setting sft parameters
        trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            #max_length=args.tok_max_length,
            #dataset_text_field="text",
            processing_class=tokenizer,
            data_collator=data_collator,
            args=training_arguments,
            #optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(args.patience)]
        )

        model.config.use_cache = False
        try:
            train_result = trainer.train()
        except KeyboardInterrupt:
            print("Caught Ctrl+C or SIGINT. Interrupting pre-training and proceeding to middle testing.")

        # Cleanup checkpoints
        print("Cleaning up checkpoints.")
        for filename in os.listdir(MODEL_SAVE_PATH + "/"):
            if filename.startswith("checkpoint-"):
                dirname = os.path.join(MODEL_SAVE_PATH, filename)
                print(f"Removing: {dirname}")
                shutil.rmtree(dirname)

        wandb.finish()

