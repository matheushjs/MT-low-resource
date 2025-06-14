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
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy import stats
import lang2vec.lang2vec as l2v
from scipy.spatial.distance import cosine

np.set_printoptions(linewidth=160)

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

FLORES_TO_PART3 = {'bam_Latn':'bam','vec_Latn':'vec','prs_Arab':'prs','tuk_Latn':'tuk','tgl_Latn':'tgl','fij_Latn':'fij','nld_Latn':'nld','luo_Latn':'luo','arb_Latn':'arb','run_Latn':'run','eus_Latn':'eus','lim_Latn':'lim','kan_Knda':'kan','ydd_Hebr':'ydd','bel_Cyrl':'bel','kik_Latn':'kik','mlt_Latn':'mlt','kmb_Latn':'kmb','mar_Deva':'mar','min_Latn':'min','umb_Latn':'umb','kas_Arab':'kas','ewe_Latn':'ewe','lmo_Latn':'lmo','wol_Latn':'wol','tam_Taml':'tam','fao_Latn':'fao','slk_Latn':'slk','deu_Latn':'deu','mya_Mymr':'mya','awa_Deva':'awa','kor_Hang':'kor','apc_Arab':'apc','hne_Deva':'hne','mkd_Cyrl':'mkd','kat_Geor':'kat','rus_Cyrl':'rus','aeb_Arab':'aeb','bjn_Arab':'bjn','hat_Latn':'hat','hau_Latn':'hau','glg_Latn':'glg','swe_Latn':'swe','mni_Beng':'mni','ben_Beng':'ben','fuv_Latn':'fuv','pap_Latn':'pap','mag_Deva':'mag','vie_Latn':'vie','twi_Latn':'twi','ces_Latn':'ces','lvs_Latn':'lvs','shn_Mymr':'shn','est_Latn':'est','kac_Latn':'kac','kab_Latn':'kab','sag_Latn':'sag','swh_Latn':'swh','ary_Arab':'ary','acq_Arab':'acq','oci_Latn':'oci','dik_Latn':'dik','ars_Arab':'ars','mal_Mlym':'mal','gla_Latn':'gla','kea_Latn':'kea','hin_Deva':'hin','tpi_Latn':'tpi','srp_Cyrl':'srp','lit_Latn':'lit','crh_Latn':'crh','tha_Thai':'tha','azj_Latn':'azj','tur_Latn':'tur','bug_Latn':'bug','por_Latn':'por','azb_Arab':'azb','tir_Ethi':'tir','ukr_Cyrl':'ukr','tum_Latn':'tum','pol_Latn':'pol','epo_Latn':'epo','ace_Arab':'ace','nso_Latn':'nso','dan_Latn':'dan','tel_Telu':'tel','scn_Latn':'scn','lij_Latn':'lij','ltz_Latn':'ltz','ilo_Latn':'ilo','dzo_Tibt':'dzo','heb_Hebr':'heb','quy_Latn':'quy','bho_Deva':'bho','knc_Arab':'knc','pes_Arab':'pes','gle_Latn':'gle','szl_Latn':'szl','zho_Hans':'zho','kam_Latn':'kam','bjn_Latn':'bjn','lus_Latn':'lus','pbt_Arab':'pbt','spa_Latn':'spa','som_Latn':'som','hye_Armn':'hye','hun_Latn':'hun','lao_Laoo':'lao','dyu_Latn':'dyu','zul_Latn':'zul','nno_Latn':'nno','ron_Latn':'ron','san_Deva':'san','xho_Latn':'xho','hrv_Latn':'hrv','lug_Latn':'lug','bem_Latn':'bem','grn_Latn':'grn','khm_Khmr':'khm','taq_Latn':'taq','urd_Arab':'urd','slv_Latn':'slv','pag_Latn':'pag','zho_Hant':'zho','tzm_Tfng':'tzm','nob_Latn':'nob','gaz_Latn':'gaz','isl_Latn':'isl','arb_Arab':'arb','sun_Latn':'sun','mri_Latn':'mri','asm_Beng':'asm','yue_Hant':'yue','ita_Latn':'ita','ban_Latn':'ban','zsm_Latn':'zsm','bod_Tibt':'bod','ibo_Latn':'ibo','kaz_Cyrl':'kaz','jpn_Jpan':'jpn','smo_Latn':'smo','npi_Deva':'npi','ckb_Arab':'ckb','afr_Latn':'afr','mos_Latn':'mos','min_Arab':'min','lin_Latn':'lin','cym_Latn':'cym','sna_Latn':'sna','ory_Orya':'ory','kin_Latn':'kin','acm_Arab':'acm','kas_Deva':'kas','lua_Latn':'lua','kbp_Latn':'kbp','ace_Latn':'ace','ceb_Latn':'ceb','als_Latn':'als','fra_Latn':'fra','nus_Latn':'nus','guj_Gujr':'guj','snd_Arab':'snd','jav_Latn':'jav','tgk_Cyrl':'tgk','ell_Grek':'ell','bak_Cyrl':'bak','ayr_Latn':'ayr','uig_Arab':'uig','sin_Sinh':'sin','ast_Latn':'ast','srd_Latn':'srd','sot_Latn':'sot','fin_Latn':'fin','tat_Cyrl':'tat','knc_Latn':'knc','nya_Latn':'nya','ssw_Latn':'ssw','yor_Latn':'yor','pan_Guru':'pan','bul_Cyrl':'bul','kir_Cyrl':'kir','bos_Latn':'bos','aka_Latn':'aka','fur_Latn':'fur','ltg_Latn':'ltg','amh_Ethi':'amh','kon_Latn':'kon','kmr_Latn':'kmr','uzn_Latn':'uzn','ind_Latn':'ind','mai_Deva':'mai','eng_Latn':'eng','plt_Latn':'plt','arz_Arab':'arz','taq_Tfng':'taq','tsn_Latn':'tsn','war_Latn':'war','tso_Latn':'tso','ajp_Arab':'ajp','khk_Cyrl':'khk','fon_Latn':'fon','cjk_Latn':'cjk','sat_Olck':'sat','cat_Latn':'cat'}
FLORES_TO_PART3['azj_Latn'] = 'aze'
FLORES_TO_PART1 = {'bam_Latn':'bm','vec_Latn':None,'prs_Arab':None,'tuk_Latn':'tk','tgl_Latn':'tl','fij_Latn':'fj','nld_Latn':'nl','luo_Latn':None,'arb_Latn':None,'run_Latn':'rn','eus_Latn':'eu','lim_Latn':'li','kan_Knda':'kn','ydd_Hebr':None,'bel_Cyrl':'be','kik_Latn':'ki','mlt_Latn':'mt','kmb_Latn':None,'mar_Deva':'mr','min_Latn':None,'umb_Latn':None,'kas_Arab':'ks','ewe_Latn':'ee','lmo_Latn':None,'wol_Latn':'wo','tam_Taml':'ta','fao_Latn':'fo','slk_Latn':'sk','deu_Latn':'de','mya_Mymr':'my','awa_Deva':None,'kor_Hang':'ko','apc_Arab':None,'hne_Deva':None,'mkd_Cyrl':'mk','kat_Geor':'ka','rus_Cyrl':'ru','aeb_Arab':None,'bjn_Arab':None,'hat_Latn':'ht','hau_Latn':'ha','glg_Latn':'gl','swe_Latn':'sv','mni_Beng':None,'ben_Beng':'bn','fuv_Latn':None,'pap_Latn':None,'mag_Deva':None,'vie_Latn':'vi','twi_Latn':'tw','ces_Latn':'cs','lvs_Latn':None,'shn_Mymr':None,'est_Latn':'et','kac_Latn':None,'kab_Latn':None,'sag_Latn':'sg','swh_Latn':None,'ary_Arab':None,'acq_Arab':None,'oci_Latn':'oc','dik_Latn':None,'ars_Arab':None,'mal_Mlym':'ml','gla_Latn':'gd','kea_Latn':None,'hin_Deva':'hi','tpi_Latn':None,'srp_Cyrl':'sr','lit_Latn':'lt','crh_Latn':None,'tha_Thai':'th','azj_Latn':'az','tur_Latn':'tr','bug_Latn':None,'por_Latn':'pt','azb_Arab':None,'tir_Ethi':'ti','ukr_Cyrl':'uk','tum_Latn':None,'pol_Latn':'pl','epo_Latn':'eo','ace_Arab':None,'nso_Latn':None,'dan_Latn':'da','tel_Telu':'te','scn_Latn':None,'lij_Latn':None,'ltz_Latn':'lb','ilo_Latn':None,'dzo_Tibt':'dz','heb_Hebr':'he','quy_Latn':None,'bho_Deva':None,'knc_Arab':None,'pes_Arab':None,'gle_Latn':'ga','szl_Latn':None,'zho_Hans':'zh','kam_Latn':None,'bjn_Latn':None,'lus_Latn':None,'pbt_Arab':None,'spa_Latn':'es','som_Latn':'so','hye_Armn':'hy','hun_Latn':'hu','lao_Laoo':'lo','dyu_Latn':None,'zul_Latn':'zu','nno_Latn':'nn','ron_Latn':'ro','san_Deva':'sa','xho_Latn':'xh','hrv_Latn':'hr','lug_Latn':'lg','bem_Latn':None,'grn_Latn':'gn','khm_Khmr':'km','taq_Latn':None,'urd_Arab':'ur','slv_Latn':'sl','pag_Latn':None,'zho_Hant':'zh','tzm_Tfng':None,'nob_Latn':'nb','gaz_Latn':None,'isl_Latn':'is','arb_Arab':'ar','sun_Latn':'su','mri_Latn':'mi','asm_Beng':'as','yue_Hant':None,'ita_Latn':'it','ban_Latn':None,'zsm_Latn':None,'bod_Tibt':'bo','ibo_Latn':'ig','kaz_Cyrl':'kk','jpn_Jpan':'ja','smo_Latn':'sm','npi_Deva':None,'ckb_Arab':None,'afr_Latn':'af','mos_Latn':None,'min_Arab':None,'lin_Latn':'ln','cym_Latn':'cy','sna_Latn':'sn','ory_Orya':None,'kin_Latn':'rw','acm_Arab':None,'kas_Deva':'ks','lua_Latn':None,'kbp_Latn':None,'ace_Latn':None,'ceb_Latn':None,'als_Latn':None,'fra_Latn':'fr','nus_Latn':None,'guj_Gujr':'gu','snd_Arab':'sd','jav_Latn':'jv','tgk_Cyrl':'tg','ell_Grek':'el','bak_Cyrl':'ba','ayr_Latn':None,'uig_Arab':'ug','sin_Sinh':'si','ast_Latn':None,'srd_Latn':'sc','sot_Latn':'st','fin_Latn':'fi','tat_Cyrl':'tt','knc_Latn':None,'nya_Latn':'ny','ssw_Latn':'ss','yor_Latn':'yo','pan_Guru':'pa','bul_Cyrl':'bg','kir_Cyrl':'ky','bos_Latn':'bs','aka_Latn':'ak','fur_Latn':None,'ltg_Latn':None,'amh_Ethi':'am','kon_Latn':'kg','kmr_Latn':None,'uzn_Latn':None,'ind_Latn':'id','mai_Deva':None,'eng_Latn':'en','plt_Latn':None,'arz_Arab':None,'taq_Tfng':None,'tsn_Latn':'tn','war_Latn':None,'tso_Latn':'ts','ajp_Arab':None,'khk_Cyrl':None,'fon_Latn':None,'cjk_Latn':None,'sat_Olck':None,'cat_Latn':'ca'}
PART3_TO_FLORES = { j: i for i,j in FLORES_TO_PART3.items() }
PART1_TO_FLORES = { j: i for i,j in FLORES_TO_PART1.items() }

# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Requires data1 and data2 to be matrices where each row is a vector
def editDistance2(data1, data2, shift_lambda=1, shift_exponent=1):
    emb_dists = distance_matrix(data1, data2)
    #print(emb_dists.shape)

    A = data1
    B = data2
    shift_dists = np.array([ [ abs( ((len(A)-1)/(len(B)-1))*j - i)**shift_exponent for j in range(len(B)) ] for i in range(len(A)) ])

    # Normalize distances
    emb_dists = ( emb_dists - np.mean(emb_dists) ) / np.std(emb_dists)
    shift_dists = ( shift_dists - np.mean(shift_dists) ) / np.std(shift_dists)

    total_dists = emb_dists + shift_lambda*shift_dists

    row_ind, col_ind = linear_sum_assignment(total_dists)

    #print(row_ind, col_ind)

    dist = total_dists[row_ind,col_ind].sum()

    return stats.norm.cdf(dist, scale=2**2)

checkpoint = "facebook/nllb-200-3.3B"
tokenizer = None
weights_array = np.load("nllb-embedding-layer.npy")
def cltad_distance(lang1, lang2, shift_lambda=1, shift_exponent=1):
    """Calculates CLTAD distance between lang1 and lang1 (2-letter language codes)"""
    
    global tokenizer
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    flores_lang1 = PART1_TO_FLORES[lang1]
    flores_lang2 = PART1_TO_FLORES[lang2]
    
    file1 = f"flores200_dataset/dev/{flores_lang1}.dev"
    file2 = f"flores200_dataset/dev/{flores_lang2}.dev"

    with open(file1) as fp:
        lines1 = fp.read().strip().split("\n")

    with open(file2) as fp:
        lines2 = fp.read().strip().split("\n")

    lines1 = lines1[:1000]
    lines2 = lines2[:1000]

    dists = []
    for l1, l2 in zip(lines1, lines2):
        tokenizer.src_lang = flores_lang1
        x = tokenizer(l1, return_tensors='pt', padding=False, truncation=False, max_length=None, add_special_tokens=False)

        tokenizer.src_lang = flores_lang2
        y = tokenizer(l2, return_tensors='pt', padding=False, truncation=False, max_length=None, add_special_tokens=False)

        x_emb = []
        for xx in x["input_ids"].numpy().ravel():
            x_emb.append(weights_array[xx,:])
        x_emb = np.array(x_emb)

        y_emb = []
        for yy in y["input_ids"].numpy().ravel():
            y_emb.append(weights_array[yy,:])
        y_emb = np.array(y_emb)

        dists.append(editDistance2(x_emb, y_emb, shift_lambda=shift_lambda, shift_exponent=shift_exponent))

    dist = np.mean(dists)
    return dist

cltad_cached = pd.read_csv("cltad-distances-reduced.csv", header=0, index_col=0)
def cltad_distance_cached(lang1, lang2):
    return cltad_cached.loc[lang1, lang2]

df_laser = pd.read_csv("laser-distances-reduced-full.csv", header=0, index_col=0)
def laser_distance(lang1, lang2):
    return df_laser.loc[PART1_TO_FLORES[lang1], PART1_TO_FLORES[lang2]]

df_eling = pd.read_csv("elinguistics-distances.csv", header=0, index_col=0)
def elinguistics_distance(lang1, lang2):
    return df_eling.loc[lang1, lang2]

l2v_processed_features = None
l2v_processed_featNames = None
def l2v_distance(lang1, lang2, lambdas=np.array([1]*33), verbose=False):
    global l2v_processed_features, l2v_processed_featNames

    data = np.load("lang2vec/lang2vec/data/features.npz")
    s_idx = np.argwhere([ i.startswith("S_") for i in data["feats"] ]).ravel()
    featNames = data["feats"][s_idx]

    lang_idx = lambda lang: list(data["langs"]).index(lang)
    
    # We consider only the sources: ETHNO, WALS and SSWL
    sources_idx = [0, 1, 7]

    get_features = lambda lang: data["data"][lang_idx(lang)][s_idx][:,sources_idx]

    langs_considered = ["hy", "az", "ka", "cs", "ro", "ru",
                        "hu", "tr", "ko", "fr", "es", "ar",
                        "he", "ja"]
    if l2v_processed_features == None:
        l2v_processed_features = dict()
        for lang in langs_considered:
            feats = get_features(FLORES_TO_PART3[PART1_TO_FLORES[lang]])
            newFeats = []
            for i in range(feats.shape[0]):
                line = feats[i,:]
                line = line[line != -1]
                if len(line) > 0:
                    newFeats.append(np.mean(line))
                else:
                    newFeats.append(-np.inf)
            l2v_processed_features[lang] = np.array(newFeats)
    
        nan_idxs = np.array([False] * len(featNames))
        for k in l2v_processed_features.keys():
            #print(k, l2v_processed_features[k])
            nan_idxs = nan_idxs | np.array(l2v_processed_features[k] < 0)

        # print(len(featNames))
        l2v_processed_featNames = featNames[np.invert(nan_idxs)]
        #print(len(featNames) - sum(nan_idxs))
        #print(featNames[np.invert(nan_idxs)])

        for k in l2v_processed_features.keys():
            l2v_processed_features[k] = l2v_processed_features[k][np.invert(nan_idxs)]
        #print(l2v_processed_features)

    if verbose:
        print("Lambdas used:")
        for feat, lamb in zip(l2v_processed_featNames, lambdas):
            print(f"{feat}: {lamb:.4f}")

    return np.mean(lambdas*np.abs(l2v_processed_features[lang1] - l2v_processed_features[lang2]))

if __name__ == "__main__":
    # print(cltad_distance("hy", "cs"))

    main = ["hy", "az", "ka", "be", "gl"]
    supp = "cs ro ru hu tr hr ko es fr he ar ja".split(" ")

    print("CLTAD distances")
    for m in main:
        dists = []
        for s in supp:
            dists.append(np.log(cltad_distance_cached(m, s)))
        
        idx = np.argsort(dists).ravel()
        print(f"Main = {m}, ",
            f"best = {[ supp[i] for i in idx[:5] ]}, ",
            "dists = " + ",".join([ "{:.1f}".format(dists[i]) for i in idx[:5] ]))

    print("\nLASER distances")
    for m in main:
        dists = []
        for s in supp:
            dists.append(laser_distance(m, s))
        
        idx = np.argsort(dists).ravel()
        print(f"Main = {m}, ",
            f"best = {[ supp[i] for i in idx[:5] ]}, ",
            "dists = " + ",".join([ "{:.3f}".format(dists[i]) for i in idx[:5] ]))

    print("\nL2v custom syntax distances")
    lambdas = np.array([0.03011268, 0., 0.61736057, 0., 0.4757546, 0.07808931, 0.001, 0.17031381, 0.03011268, 0., 0.001, 0.001, 0., 0., 0., 0., 0.001, 0.40010741, 0., 0.4006355, 0.40010603, 0.69080728, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0., 0., 0.58323313])
    for m in main:
        dists = []
        for s in supp:
            try:
                dists.append(l2v_distance(m, s, lambdas=lambdas))
            except:
                dists.append(10)
        
        idx = np.argsort(dists).ravel()
        print(f"Main = {m}, ",
            f"best = {[ supp[i] for i in idx[:5] ]}, ",
            "dists = " + ",".join([ "{:.3f}".format(dists[i]) for i in idx[:5] ]))

    print("\nL2v genetic distances")
    for m in main:
        dists = []
        for s in supp:
            dists.append(l2v.genetic_distance(FLORES_TO_PART3[PART1_TO_FLORES[m]], FLORES_TO_PART3[PART1_TO_FLORES[s]]))

        idx = np.argsort(dists).ravel()
        print(f"Main = {m}, ",
            f"best = {[ supp[i] for i in idx[:5] ]}, ",
            "dists = " + ",".join([ "{:.3f}".format(dists[i]) for i in idx[:5] ]))
        
    print("\nElinguistics genetic distances")
    for m in main:
        dists = []
        for s in supp:
            dists.append(elinguistics_distance(m, s))

        idx = np.argsort(dists).ravel()
        print(f"Main = {m}, ",
            f"best = {[ supp[i] for i in idx[:5] ]}, ",
            "dists = " + ",".join([ "{:.3f}".format(dists[i]) for i in idx[:5] ]))