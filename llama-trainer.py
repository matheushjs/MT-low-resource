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
parser.add_argument("--patience",
        help="Patience of early stopping. Applied to both pre- and post-training.",
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
        default=5e-4)
parser.add_argument("--post-learning-rate",
        help="Learning rate for post-training.",
        type=float,
        default=1e-5)
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
        default=20)
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
FLORES_TO_PART1 = {'bam_Latn':'bm','vec_Latn':None,'prs_Arab':None,'tuk_Latn':'tk','tgl_Latn':'tl','fij_Latn':'fj','nld_Latn':'nl','luo_Latn':None,'arb_Latn':None,'run_Latn':'rn','eus_Latn':'eu','lim_Latn':'li','kan_Knda':'kn','ydd_Hebr':None,'bel_Cyrl':'be','kik_Latn':'ki','mlt_Latn':'mt','kmb_Latn':None,'mar_Deva':'mr','min_Latn':None,'umb_Latn':None,'kas_Arab':'ks','ewe_Latn':'ee','lmo_Latn':None,'wol_Latn':'wo','tam_Taml':'ta','fao_Latn':'fo','slk_Latn':'sk','deu_Latn':'de','mya_Mymr':'my','awa_Deva':None,'kor_Hang':'ko','apc_Arab':None,'hne_Deva':None,'mkd_Cyrl':'mk','kat_Geor':'ka','rus_Cyrl':'ru','aeb_Arab':None,'bjn_Arab':None,'hat_Latn':'ht','hau_Latn':'ha','glg_Latn':'gl','swe_Latn':'sv','mni_Beng':None,'ben_Beng':'bn','fuv_Latn':None,'pap_Latn':None,'mag_Deva':None,'vie_Latn':'vi','twi_Latn':'tw','ces_Latn':'cs','lvs_Latn':None,'shn_Mymr':None,'est_Latn':'et','kac_Latn':None,'kab_Latn':None,'sag_Latn':'sg','swh_Latn':None,'ary_Arab':None,'acq_Arab':None,'oci_Latn':'oc','dik_Latn':None,'ars_Arab':None,'mal_Mlym':'ml','gla_Latn':'gd','kea_Latn':None,'hin_Deva':'hi','tpi_Latn':None,'srp_Cyrl':'sr','lit_Latn':'lt','crh_Latn':None,'tha_Thai':'th','azj_Latn':None,'tur_Latn':'tr','bug_Latn':None,'por_Latn':'pt','azb_Arab':None,'tir_Ethi':'ti','ukr_Cyrl':'uk','tum_Latn':None,'pol_Latn':'pl','epo_Latn':'eo','ace_Arab':None,'nso_Latn':None,'dan_Latn':'da','tel_Telu':'te','scn_Latn':None,'lij_Latn':None,'ltz_Latn':'lb','ilo_Latn':None,'dzo_Tibt':'dz','heb_Hebr':'he','quy_Latn':None,'bho_Deva':None,'knc_Arab':None,'pes_Arab':None,'gle_Latn':'ga','szl_Latn':None,'zho_Hans':'zh','kam_Latn':None,'bjn_Latn':None,'lus_Latn':None,'pbt_Arab':None,'spa_Latn':'es','som_Latn':'so','hye_Armn':'hy','hun_Latn':'hu','lao_Laoo':'lo','dyu_Latn':None,'zul_Latn':'zu','nno_Latn':'nn','ron_Latn':'ro','san_Deva':'sa','xho_Latn':'xh','hrv_Latn':'hr','lug_Latn':'lg','bem_Latn':None,'grn_Latn':'gn','khm_Khmr':'km','taq_Latn':None,'urd_Arab':'ur','slv_Latn':'sl','pag_Latn':None,'zho_Hant':'zh','tzm_Tfng':None,'nob_Latn':'nb','gaz_Latn':None,'isl_Latn':'is','arb_Arab':None,'sun_Latn':'su','mri_Latn':'mi','asm_Beng':'as','yue_Hant':None,'ita_Latn':'it','ban_Latn':None,'zsm_Latn':None,'bod_Tibt':'bo','ibo_Latn':'ig','kaz_Cyrl':'kk','jpn_Jpan':'ja','smo_Latn':'sm','npi_Deva':None,'ckb_Arab':None,'afr_Latn':'af','mos_Latn':None,'min_Arab':None,'lin_Latn':'ln','cym_Latn':'cy','sna_Latn':'sn','ory_Orya':None,'kin_Latn':'rw','acm_Arab':None,'kas_Deva':'ks','lua_Latn':None,'kbp_Latn':None,'ace_Latn':None,'ceb_Latn':None,'als_Latn':None,'fra_Latn':'fr','nus_Latn':None,'guj_Gujr':'gu','snd_Arab':'sd','jav_Latn':'jv','tgk_Cyrl':'tg','ell_Grek':'el','bak_Cyrl':'ba','ayr_Latn':None,'uig_Arab':'ug','sin_Sinh':'si','ast_Latn':None,'srd_Latn':'sc','sot_Latn':'st','fin_Latn':'fi','tat_Cyrl':'tt','knc_Latn':None,'nya_Latn':'ny','ssw_Latn':'ss','yor_Latn':'yo','pan_Guru':'pa','bul_Cyrl':'bg','kir_Cyrl':'ky','bos_Latn':'bs','aka_Latn':'ak','fur_Latn':None,'ltg_Latn':None,'amh_Ethi':'am','kon_Latn':'kg','kmr_Latn':None,'uzn_Latn':None,'ind_Latn':'id','mai_Deva':None,'eng_Latn':'en','plt_Latn':None,'arz_Arab':None,'taq_Tfng':None,'tsn_Latn':'tn','war_Latn':None,'tso_Latn':'ts','ajp_Arab':None,'khk_Cyrl':None,'fon_Latn':None,'cjk_Latn':None,'sat_Olck':None,'cat_Latn':'ca'}
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

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def dataset_for_llama(tokenizer, lang_pairs=args.lang_pairs):
    """Generate Dataset for Llama or Qwen"""
    
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
        
        # if args.limit_train_corpus > 0 and len(new_data["train"]) > args.limit_train_corpus:
        #     new_data["train"] = new_data["train"].shuffle().select(range(args.limit_train_corpus))

        # TODO: implement this properly (with a custom Dataset subclass)
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

    def _format_template(row):
        instruction = f"You are a professional translator proficient in translating {row["name1"]} text to {row["name2"]}. " + \
                       "Your responses should contain only the translated text without any additional commentary."
        prompt = f"Translate this from {row["name1"]} to {row["name2"]}: {row["sentence1"]}"
        row_json = [{"role": "system", "content": instruction },
                    {"role": "user", "content": prompt },
                    {"role": "assistant", "content": row["sentence2"] } ]
        text = tokenizer.apply_chat_template(row_json, tokenize=False)

        row["instruction"] = instruction
        row["response"] = row["sentence2"]
        row["text"] = text

        return row

    return new_dataset.map(_format_template, num_proc=4).shuffle()

def translate(
    text, tokenizer, model, src_lang, tgt_lang, 
    a=32, b=3, max_input_length=256, num_beams=1, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""
    
    instruction = f"You are a professional translator proficient in translating {src_lang} text to {tgt_lang}. \
    Your responses should contain only the translated text without any additional commentary."
    prompt = f"Translate this from {src_lang} to {tgt_lang}: {text}"
    chat_style_prompt = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
    prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=max_input_length, truncation=True)
    model.eval()
    result = model.generate(
        **inputs.to(model.device),
        num_beams=num_beams,
        max_new_tokens=int(a + b * len(text)),
        do_sample=False, top_p=None, temperature=None
    )
    text = tokenizer.decode(result[0], skip_special_tokens=True)
    return "".join(text.split("assistant")[1:]).strip()

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

if __name__ == "__main__":
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
            quantization_config=bnb_config,
            device_map="auto"
        )

    else:
        print("Loading model in float16/bfloat16 precision.")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype,
            device_map="auto"
        )

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
    translations = []
    training_step_counter = 0
    post_training_step_counter = 0
    time_begin = time.time()

    datasets = dataset_for_llama(tokenizer, lang_pairs=args.lang_pairs)

    train_dataset = datasets["train"]
    post_train_dataset = datasets["train"].shuffle().filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
    dev_dataset = datasets["dev"].filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)
    test_dataset = datasets["test"].filter(lambda x: x["lang1"] == main_lang1 and x["lang2"] == main_lang2)

    print("Printing some samples of the Llama dataset.")
    print("lang1: ", "|".join(train_dataset["lang1"][:30]))
    print("lang2: ", "|".join(train_dataset["lang2"][:30]))
    print("name1: ", "|".join(train_dataset["name1"][:30]))
    print("name2: ", "|".join(train_dataset["name2"][:30]))
    print("sentence1:\n", "\n".join(train_dataset["sentence1"][:5]))
    print("sentence2:\n", "\n".join(train_dataset["sentence2"][:5]))
    print("instruction:\n", "\n".join(train_dataset["instruction"][:5]))
    print("text:\n", "\n".join(train_dataset["text"][:5]))

    print("Printing some samples of the Evaluation dataset.")
    print("lang1: ", "|".join(dev_dataset["lang1"][:30]))
    print("lang2: ", "|".join(dev_dataset["lang2"][:30]))
    print("sentence1:\n", "\n".join(dev_dataset["sentence1"][:5]))
    print("sentence2:\n", "\n".join(dev_dataset["sentence2"][:5]))
    print("instruction:\n", "\n".join(dev_dataset["instruction"][:5]))
    print("text:\n", "\n".join(dev_dataset["text"][:5]))

    print(f"Dataset sizes:\ntrain: {len(train_dataset)}\npost_train: {len(post_train_dataset)}\ndev: {len(dev_dataset)}\ntest: {len(test_dataset)}")

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
        try:
            model, tokenizer = setup_chat_format(model, tokenizer)
        except Exception as e:
            print(e)
        model = get_peft_model(model, peft_config)

        run = wandb.init(
            entity="elfmathews-university-of-tsukuba",
            project=EXPERIMENT_NAME, 
            job_type="training", 
            anonymous="allow"
        )

        #Hyperparamter
        training_arguments = SFTConfig(
            output_dir=MODEL_SAVE_PATH,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_steps=1,
            logging_strategy="steps",
            learning_rate=args.learning_rate if len(args.lang_pairs) > 1 else args.post_learning_rate,
            fp16=False,
            bf16=False,
            weight_decay=args.weight_decay,
            group_by_length=False,
            max_seq_length=args.tok_max_length,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="constant",
            dataloader_num_workers=4,
            dataset_text_field="text", # This argument was on Trainer
            packing=False, # This argument was on Trainer
            #max_length=512, # Idk what's going on. This is the latest version of TRL but it doesn't seem like it.
            report_to="wandb"
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
            callbacks=[EarlyStoppingCallback(args.patience)]
        )

        model.config.use_cache = False
        train_result = trainer.train()
