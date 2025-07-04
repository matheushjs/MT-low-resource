import transformers, datasets, torch
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys, os, re, time, unicodedata, pickle, html, copy, shutil
import sacrebleu
from iso639 import Language
from pathlib import Path
from sacremoses import MosesPunctNormalizer
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    logging, EarlyStoppingCallback, DataCollatorForSeq2Seq
)
from transformers.optimization import Adafactor
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime as dt
from torch.utils.data import DataLoader, Dataset as torchDataset
import seaborn as sns

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
        help="Learning rate for post-training.",
        type=float,
        default=1e-4)
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
FLORES_TO_PART1 = {'bam_Latn':'bm','vec_Latn':None,'prs_Arab':None,'tuk_Latn':'tk','tgl_Latn':'tl','fij_Latn':'fj','nld_Latn':'nl','luo_Latn':None,'arb_Latn':None,'run_Latn':'rn','eus_Latn':'eu','lim_Latn':'li','kan_Knda':'kn','ydd_Hebr':None,'bel_Cyrl':'be','kik_Latn':'ki','mlt_Latn':'mt','kmb_Latn':None,'mar_Deva':'mr','min_Latn':None,'umb_Latn':None,'kas_Arab':'ks','ewe_Latn':'ee','lmo_Latn':None,'wol_Latn':'wo','tam_Taml':'ta','fao_Latn':'fo','slk_Latn':'sk','deu_Latn':'de','mya_Mymr':'my','awa_Deva':None,'kor_Hang':'ko','apc_Arab':None,'hne_Deva':None,'mkd_Cyrl':'mk','kat_Geor':'ka','rus_Cyrl':'ru','aeb_Arab':None,'bjn_Arab':None,'hat_Latn':'ht','hau_Latn':'ha','glg_Latn':'gl','swe_Latn':'sv','mni_Beng':None,'ben_Beng':'bn','fuv_Latn':None,'pap_Latn':None,'mag_Deva':None,'vie_Latn':'vi','twi_Latn':'tw','ces_Latn':'cs','lvs_Latn':None,'shn_Mymr':None,'est_Latn':'et','kac_Latn':None,'kab_Latn':None,'sag_Latn':'sg','swh_Latn':None,'ary_Arab':None,'acq_Arab':None,'oci_Latn':'oc','dik_Latn':None,'ars_Arab':None,'mal_Mlym':'ml','gla_Latn':'gd','kea_Latn':None,'hin_Deva':'hi','tpi_Latn':None,'srp_Cyrl':'sr','lit_Latn':'lt','crh_Latn':None,'tha_Thai':'th','azj_Latn':'az','tur_Latn':'tr','bug_Latn':None,'por_Latn':'pt','azb_Arab':None,'tir_Ethi':'ti','ukr_Cyrl':'uk','tum_Latn':None,'pol_Latn':'pl','epo_Latn':'eo','ace_Arab':None,'nso_Latn':None,'dan_Latn':'da','tel_Telu':'te','scn_Latn':None,'lij_Latn':None,'ltz_Latn':'lb','ilo_Latn':None,'dzo_Tibt':'dz','heb_Hebr':'he','quy_Latn':None,'bho_Deva':None,'knc_Arab':None,'pes_Arab':None,'gle_Latn':'ga','szl_Latn':None,'zho_Hans':'zh','kam_Latn':None,'bjn_Latn':None,'lus_Latn':None,'pbt_Arab':None,'spa_Latn':'es','som_Latn':'so','hye_Armn':'hy','hun_Latn':'hu','lao_Laoo':'lo','dyu_Latn':None,'zul_Latn':'zu','nno_Latn':'nn','ron_Latn':'ro','san_Deva':'sa','xho_Latn':'xh','hrv_Latn':'hr','lug_Latn':'lg','bem_Latn':None,'grn_Latn':'gn','khm_Khmr':'km','taq_Latn':None,'urd_Arab':'ur','slv_Latn':'sl','pag_Latn':None,'zho_Hant':'zh','tzm_Tfng':None,'nob_Latn':'nb','gaz_Latn':None,'isl_Latn':'is','arb_Arab':None,'sun_Latn':'su','mri_Latn':'mi','asm_Beng':'as','yue_Hant':None,'ita_Latn':'it','ban_Latn':None,'zsm_Latn':None,'bod_Tibt':'bo','ibo_Latn':'ig','kaz_Cyrl':'kk','jpn_Jpan':'ja','smo_Latn':'sm','npi_Deva':None,'ckb_Arab':None,'afr_Latn':'af','mos_Latn':None,'min_Arab':None,'lin_Latn':'ln','cym_Latn':'cy','sna_Latn':'sn','ory_Orya':None,'kin_Latn':'rw','acm_Arab':None,'kas_Deva':'ks','lua_Latn':None,'kbp_Latn':None,'ace_Latn':None,'ceb_Latn':None,'als_Latn':None,'fra_Latn':'fr','nus_Latn':None,'guj_Gujr':'gu','snd_Arab':'sd','jav_Latn':'jv','tgk_Cyrl':'tg','ell_Grek':'el','bak_Cyrl':'ba','ayr_Latn':None,'uig_Arab':'ug','sin_Sinh':'si','ast_Latn':None,'srd_Latn':'sc','sot_Latn':'st','fin_Latn':'fi','tat_Cyrl':'tt','knc_Latn':None,'nya_Latn':'ny','ssw_Latn':'ss','yor_Latn':'yo','pan_Guru':'pa','bul_Cyrl':'bg','kir_Cyrl':'ky','bos_Latn':'bs','aka_Latn':'ak','fur_Latn':None,'ltg_Latn':None,'amh_Ethi':'am','kon_Latn':'kg','kmr_Latn':None,'uzn_Latn':None,'ind_Latn':'id','mai_Deva':None,'eng_Latn':'en','plt_Latn':None,'arz_Arab':None,'taq_Tfng':None,'tsn_Latn':'tn','war_Latn':None,'tso_Latn':'ts','ajp_Arab':None,'khk_Cyrl':None,'fon_Latn':None,'cjk_Latn':None,'sat_Olck':None,'cat_Latn':'ca'}
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

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def dataset_for_nllb(tokenizer, lang_pairs=args.lang_pairs):
    dataset = load_dataset("json", data_files={
        "train": "./ted-multiling/train.json",
        "test": "./ted-multiling/test.json",
        "dev": "./ted-multiling/dev.json"
    })

    paired_sentences_datasets = []

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

        # TODO: Fix this dirty plan and implement a problem Dataset class
        # For now, we truncate OR expand all corpuses to the limit_train_corpus
        if args.limit_train_corpus > 0:
            if len(new_data["train"]) > args.limit_train_corpus:
                new_data["train"] = new_data["train"].shuffle().select(range(args.limit_train_corpus))
            elif len(new_data["train"]) < args.limit_train_corpus:
                multiplier = int(np.ceil(args.limit_train_corpus / len(new_data["train"])))
                new_data["train"] = datasets.concatenate_datasets([ new_data["train"] for k in range(multiplier) ])
                new_data["train"] = new_data["train"].select(range(args.limit_train_corpus))

        print(f"Length of {lang_pair} dataset:", len(new_data["train"]), len(new_data["dev"]), len(new_data["test"]))

        paired_sentences_datasets.append(new_data)

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

if __name__ == "__main__":
    #checkpoint = "facebook/nllb-200-distilled-600M"
    
    checkpoint = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) #, padding_size="left")
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

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
            truncation=False,
            padding=True
        )
        for k in row.keys():
            model_inputs[k] = row[k]
        model_inputs["len1"] = len(model_inputs['input_ids'])
        model_inputs["len2"] = len(model_inputs['labels'])
        return model_inputs

    train_dataset      = datasets["train"] \
                            .map(_tokenize_fn, num_proc=8)

    df = train_dataset.to_pandas()

    sns.set()
    cmap = sns.color_palette("Set2")
    sns.set_palette("Set2")

    TOKENIZER = "Llama"

    data = df.query("lang1 == 'en' and lang2 == 'hy'")[["len1", "len2"]]
    hike = np.round((np.mean(data["len2"] / data["len1"]) - 1) * 100, 2)
    plt.scatter(data["len1"], data["len2"], s=5, c=cmap[0])
    x0, x1, y0, y1 = plt.axis()
    plt.plot([0, x1], [0, x1], c="black", linestyle="dashed")
    plt.title(f"Token Count ({TOKENIZER} tokenizer) -- English vs Armenian")
    plt.xlabel("English token count")
    plt.ylabel("Armenian token count")
    plt.text(0, 0.9*y1, f"{hike}% more tokens on average.")
    plt.tight_layout()
    plt.savefig(f"tokcount-{TOKENIZER}-en-hy.png", dpi=150)
    plt.show()

    data = df.query("lang1 == 'en' and lang2 == 'az'")[["len1", "len2"]]
    hike = np.round((np.mean(data["len2"] / data["len1"]) - 1) * 100, 2)
    plt.scatter(data["len1"], data["len2"], s=5, c=cmap[1])
    x0, x1, y0, y1 = plt.axis()
    plt.plot([0, x1], [0, x1], c="black", linestyle="dashed")
    plt.title(f"Token Count ({TOKENIZER} tokenizer) -- English vs Azerbaijani")
    plt.xlabel("English token count")
    plt.ylabel("Azerbaijani token count")
    plt.text(0, 0.9*y1, f"{hike}% more tokens on average.")
    plt.tight_layout()
    plt.savefig(f"tokcount-{TOKENIZER}-en-az.png", dpi=150)
    plt.show()

    data = df.query("lang1 == 'en' and lang2 == 'ka'")[["len1", "len2"]]
    hike = np.round((np.mean(data["len2"] / data["len1"]) - 1) * 100, 2)
    plt.scatter(data["len1"], data["len2"], s=5, c=cmap[2])
    x0, x1, y0, y1 = plt.axis()
    plt.plot([0, x1], [0, x1], c="black", linestyle="dashed")
    plt.title(f"Token Count ({TOKENIZER} tokenizer) -- English vs Georgian")
    plt.xlabel("English token count")
    plt.ylabel("Georgian token count")
    plt.text(0, 0.9*y1, f"{hike}% more tokens on average.")
    plt.tight_layout()
    plt.savefig(f"tokcount-{TOKENIZER}-en-ka.png", dpi=150)
    plt.show()
