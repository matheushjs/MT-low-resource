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

# Check language codes at: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
FLORES_TO_PART3 = {'bam_Latn':'bam','vec_Latn':'vec','prs_Arab':'prs','tuk_Latn':'tuk','tgl_Latn':'tgl','fij_Latn':'fij','nld_Latn':'nld','luo_Latn':'luo','arb_Latn':'arb','run_Latn':'run','eus_Latn':'eus','lim_Latn':'lim','kan_Knda':'kan','ydd_Hebr':'ydd','bel_Cyrl':'bel','kik_Latn':'kik','mlt_Latn':'mlt','kmb_Latn':'kmb','mar_Deva':'mar','min_Latn':'min','umb_Latn':'umb','kas_Arab':'kas','ewe_Latn':'ewe','lmo_Latn':'lmo','wol_Latn':'wol','tam_Taml':'tam','fao_Latn':'fao','slk_Latn':'slk','deu_Latn':'deu','mya_Mymr':'mya','awa_Deva':'awa','kor_Hang':'kor','apc_Arab':'apc','hne_Deva':'hne','mkd_Cyrl':'mkd','kat_Geor':'kat','rus_Cyrl':'rus','aeb_Arab':'aeb','bjn_Arab':'bjn','hat_Latn':'hat','hau_Latn':'hau','glg_Latn':'glg','swe_Latn':'swe','mni_Beng':'mni','ben_Beng':'ben','fuv_Latn':'fuv','pap_Latn':'pap','mag_Deva':'mag','vie_Latn':'vie','twi_Latn':'twi','ces_Latn':'ces','lvs_Latn':'lvs','shn_Mymr':'shn','est_Latn':'est','kac_Latn':'kac','kab_Latn':'kab','sag_Latn':'sag','swh_Latn':'swh','ary_Arab':'ary','acq_Arab':'acq','oci_Latn':'oci','dik_Latn':'dik','ars_Arab':'ars','mal_Mlym':'mal','gla_Latn':'gla','kea_Latn':'kea','hin_Deva':'hin','tpi_Latn':'tpi','srp_Cyrl':'srp','lit_Latn':'lit','crh_Latn':'crh','tha_Thai':'tha','azj_Latn':'azj','tur_Latn':'tur','bug_Latn':'bug','por_Latn':'por','azb_Arab':'azb','tir_Ethi':'tir','ukr_Cyrl':'ukr','tum_Latn':'tum','pol_Latn':'pol','epo_Latn':'epo','ace_Arab':'ace','nso_Latn':'nso','dan_Latn':'dan','tel_Telu':'tel','scn_Latn':'scn','lij_Latn':'lij','ltz_Latn':'ltz','ilo_Latn':'ilo','dzo_Tibt':'dzo','heb_Hebr':'heb','quy_Latn':'quy','bho_Deva':'bho','knc_Arab':'knc','pes_Arab':'pes','gle_Latn':'gle','szl_Latn':'szl','zho_Hans':'zho','kam_Latn':'kam','bjn_Latn':'bjn','lus_Latn':'lus','pbt_Arab':'pbt','spa_Latn':'spa','som_Latn':'som','hye_Armn':'hye','hun_Latn':'hun','lao_Laoo':'lao','dyu_Latn':'dyu','zul_Latn':'zul','nno_Latn':'nno','ron_Latn':'ron','san_Deva':'san','xho_Latn':'xho','hrv_Latn':'hrv','lug_Latn':'lug','bem_Latn':'bem','grn_Latn':'grn','khm_Khmr':'khm','taq_Latn':'taq','urd_Arab':'urd','slv_Latn':'slv','pag_Latn':'pag','zho_Hant':'zho','tzm_Tfng':'tzm','nob_Latn':'nob','gaz_Latn':'gaz','isl_Latn':'isl','arb_Arab':'arb','sun_Latn':'sun','mri_Latn':'mri','asm_Beng':'asm','yue_Hant':'yue','ita_Latn':'ita','ban_Latn':'ban','zsm_Latn':'zsm','bod_Tibt':'bod','ibo_Latn':'ibo','kaz_Cyrl':'kaz','jpn_Jpan':'jpn','smo_Latn':'smo','npi_Deva':'npi','ckb_Arab':'ckb','afr_Latn':'afr','mos_Latn':'mos','min_Arab':'min','lin_Latn':'lin','cym_Latn':'cym','sna_Latn':'sna','ory_Orya':'ory','kin_Latn':'kin','acm_Arab':'acm','kas_Deva':'kas','lua_Latn':'lua','kbp_Latn':'kbp','ace_Latn':'ace','ceb_Latn':'ceb','als_Latn':'als','fra_Latn':'fra','nus_Latn':'nus','guj_Gujr':'guj','snd_Arab':'snd','jav_Latn':'jav','tgk_Cyrl':'tgk','ell_Grek':'ell','bak_Cyrl':'bak','ayr_Latn':'ayr','uig_Arab':'uig','sin_Sinh':'sin','ast_Latn':'ast','srd_Latn':'srd','sot_Latn':'sot','fin_Latn':'fin','tat_Cyrl':'tat','knc_Latn':'knc','nya_Latn':'nya','ssw_Latn':'ssw','yor_Latn':'yor','pan_Guru':'pan','bul_Cyrl':'bul','kir_Cyrl':'kir','bos_Latn':'bos','aka_Latn':'aka','fur_Latn':'fur','ltg_Latn':'ltg','amh_Ethi':'amh','kon_Latn':'kon','kmr_Latn':'kmr','uzn_Latn':'uzn','ind_Latn':'ind','mai_Deva':'mai','eng_Latn':'eng','plt_Latn':'plt','arz_Arab':'arz','taq_Tfng':'taq','tsn_Latn':'tsn','war_Latn':'war','tso_Latn':'tso','ajp_Arab':'ajp','khk_Cyrl':'khk','fon_Latn':'fon','cjk_Latn':'cjk','sat_Olck':'sat','cat_Latn':'cat'}
FLORES_TO_PART1 = {'bam_Latn':'bm','vec_Latn':None,'prs_Arab':None,'tuk_Latn':'tk','tgl_Latn':'tl','fij_Latn':'fj','nld_Latn':'nl','luo_Latn':None,'arb_Latn':None,'run_Latn':'rn','eus_Latn':'eu','lim_Latn':'li','kan_Knda':'kn','ydd_Hebr':None,'bel_Cyrl':'be','kik_Latn':'ki','mlt_Latn':'mt','kmb_Latn':None,'mar_Deva':'mr','min_Latn':None,'umb_Latn':None,'kas_Arab':'ks','ewe_Latn':'ee','lmo_Latn':None,'wol_Latn':'wo','tam_Taml':'ta','fao_Latn':'fo','slk_Latn':'sk','deu_Latn':'de','mya_Mymr':'my','awa_Deva':None,'kor_Hang':'ko','apc_Arab':None,'hne_Deva':None,'mkd_Cyrl':'mk','kat_Geor':'ka','rus_Cyrl':'ru','aeb_Arab':None,'bjn_Arab':None,'hat_Latn':'ht','hau_Latn':'ha','glg_Latn':'gl','swe_Latn':'sv','mni_Beng':None,'ben_Beng':'bn','fuv_Latn':None,'pap_Latn':None,'mag_Deva':None,'vie_Latn':'vi','twi_Latn':'tw','ces_Latn':'cs','lvs_Latn':None,'shn_Mymr':None,'est_Latn':'et','kac_Latn':None,'kab_Latn':None,'sag_Latn':'sg','swh_Latn':None,'ary_Arab':None,'acq_Arab':None,'oci_Latn':'oc','dik_Latn':None,'ars_Arab':None,'mal_Mlym':'ml','gla_Latn':'gd','kea_Latn':None,'hin_Deva':'hi','tpi_Latn':None,'srp_Cyrl':'sr','lit_Latn':'lt','crh_Latn':None,'tha_Thai':'th','azj_Latn':'az','tur_Latn':'tr','bug_Latn':None,'por_Latn':'pt-br','azb_Arab':None,'tir_Ethi':'ti','ukr_Cyrl':'uk','tum_Latn':None,'pol_Latn':'pl','epo_Latn':'eo','ace_Arab':None,'nso_Latn':None,'dan_Latn':'da','tel_Telu':'te','scn_Latn':None,'lij_Latn':None,'ltz_Latn':'lb','ilo_Latn':None,'dzo_Tibt':'dz','heb_Hebr':'he','quy_Latn':None,'bho_Deva':None,'knc_Arab':None,'pes_Arab':'fa','gle_Latn':'ga','szl_Latn':None,'zho_Hans':'zh-cn','kam_Latn':None,'bjn_Latn':None,'lus_Latn':None,'pbt_Arab':None,'spa_Latn':'es','som_Latn':'so','hye_Armn':'hy','hun_Latn':'hu','lao_Laoo':'lo','dyu_Latn':None,'zul_Latn':'zu','nno_Latn':'nn','ron_Latn':'ro','san_Deva':'sa','xho_Latn':'xh','hrv_Latn':'hr','lug_Latn':'lg','bem_Latn':None,'grn_Latn':'gn','khm_Khmr':'km','taq_Latn':None,'urd_Arab':'ur','slv_Latn':'sl','pag_Latn':None,'zho_Hant':'zh-tw','tzm_Tfng':None,'nob_Latn':'nb','gaz_Latn':None,'isl_Latn':'is','arb_Arab':'ar','sun_Latn':'su','mri_Latn':'mi','asm_Beng':'as','yue_Hant':None,'ita_Latn':'it','ban_Latn':None,'zsm_Latn':'ms','bod_Tibt':'bo','ibo_Latn':'ig','kaz_Cyrl':'kk','jpn_Jpan':'ja','smo_Latn':'sm','npi_Deva':None,'ckb_Arab':None,'afr_Latn':'af','mos_Latn':None,'min_Arab':None,'lin_Latn':'ln','cym_Latn':'cy','sna_Latn':'sn','ory_Orya':None,'kin_Latn':'rw','acm_Arab':None,'kas_Deva':'ks','lua_Latn':None,'kbp_Latn':None,'ace_Latn':None,'ceb_Latn':None,'als_Latn':'sq','fra_Latn':'fr','nus_Latn':None,'guj_Gujr':'gu','snd_Arab':'sd','jav_Latn':'jv','tgk_Cyrl':'tg','ell_Grek':'el','bak_Cyrl':'ba','ayr_Latn':None,'uig_Arab':'ug','sin_Sinh':'si','ast_Latn':None,'srd_Latn':'sc','sot_Latn':'st','fin_Latn':'fi','tat_Cyrl':'tt','knc_Latn':None,'nya_Latn':'ny','ssw_Latn':'ss','yor_Latn':'yo','pan_Guru':'pa','bul_Cyrl':'bg','kir_Cyrl':'ky','bos_Latn':'bs','aka_Latn':'ak','fur_Latn':None,'ltg_Latn':None,'amh_Ethi':'am','kon_Latn':'kg','kmr_Latn':None,'uzn_Latn':None,'ind_Latn':'id','mai_Deva':None,'eng_Latn':'en','plt_Latn':None,'arz_Arab':None,'taq_Tfng':None,'tsn_Latn':'tn','war_Latn':None,'tso_Latn':'ts','ajp_Arab':None,'khk_Cyrl':'mn','fon_Latn':None,'cjk_Latn':None,'sat_Olck':None,'cat_Latn':'ca'}
PART3_TO_FLORES = { j: i for i,j in FLORES_TO_PART3.items() }
PART1_TO_FLORES = { j: i for i,j in FLORES_TO_PART1.items() }

dataset = load_dataset("json", data_files={
        "train": "./ted-multiling/train.json",
        "test": "./ted-multiling/test.json",
        "dev": "./ted-multiling/dev.json"
    })

counts = [ (k, sum([ i != "__NULL__" for i in dataset["train"][k] ])) for k in dataset["train"].column_names ]
print("Counts before:")
[ print(i) for i in counts ]

train_df = dataset["train"].to_dict()
test_df = dataset["test"].to_dict()
dev_df = dataset["dev"].to_dict()

# Let's first filter duplicates on small sentences
def filter_dup(df):
    lines = np.array(df["en"])
    shortLines = [ str(i) for i in filter(lambda x: len(x) < 15, lines) ]
    uniq = np.unique(shortLines)
    removeIdx = []

    for i in uniq:
        idx = np.argwhere(i == lines).ravel()
        if len(idx) > 1:
            # We remove all but the first occurrence
            removeIdx.extend(idx[1:])

    removeIdx = sorted(np.unique(removeIdx))
    print(f"Removing {len(removeIdx)} elements.")
    for idx in removeIdx[::-1]:
        for k in df.keys():
            df[k].pop(idx)
    
    return df

train_df = Dataset.from_dict(filter_dup(train_df))
test_df = Dataset.from_dict(filter_dup(test_df))
dev_df = Dataset.from_dict(filter_dup(dev_df))

dataset = DatasetDict({
    "train": train_df,
    "test": test_df,
    "dev": dev_df
})

counts = [ (k, sum([ i != "__NULL__" for i in dataset["train"][k] ])) for k in dataset["train"].column_names ]
print("Counts after:")
[ print(i) for i in counts ]

dataset["train"].to_json("./ted-multiling-filtered/train.json")
dataset["test"].to_json("./ted-multiling-filtered/test.json")
dataset["dev"].to_json("./ted-multiling-filtered/dev.json")