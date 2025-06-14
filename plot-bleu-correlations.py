import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iso639 import Language
from lang2vec import lang2vec as l2v
from io import StringIO
from language_distance_metrics import cltad_distance, l2v_distance, elinguistics_distance

sns.set()
cmap = sns.color_palette("Set2")

FLORES_TO_PART3 = {'bam_Latn':'bam','vec_Latn':'vec','prs_Arab':'prs','tuk_Latn':'tuk','tgl_Latn':'tgl','fij_Latn':'fij','nld_Latn':'nld','luo_Latn':'luo','arb_Latn':'arb','run_Latn':'run','eus_Latn':'eus','lim_Latn':'lim','kan_Knda':'kan','ydd_Hebr':'ydd','bel_Cyrl':'bel','kik_Latn':'kik','mlt_Latn':'mlt','kmb_Latn':'kmb','mar_Deva':'mar','min_Latn':'min','umb_Latn':'umb','kas_Arab':'kas','ewe_Latn':'ewe','lmo_Latn':'lmo','wol_Latn':'wol','tam_Taml':'tam','fao_Latn':'fao','slk_Latn':'slk','deu_Latn':'deu','mya_Mymr':'mya','awa_Deva':'awa','kor_Hang':'kor','apc_Arab':'apc','hne_Deva':'hne','mkd_Cyrl':'mkd','kat_Geor':'kat','rus_Cyrl':'rus','aeb_Arab':'aeb','bjn_Arab':'bjn','hat_Latn':'hat','hau_Latn':'hau','glg_Latn':'glg','swe_Latn':'swe','mni_Beng':'mni','ben_Beng':'ben','fuv_Latn':'fuv','pap_Latn':'pap','mag_Deva':'mag','vie_Latn':'vie','twi_Latn':'twi','ces_Latn':'ces','lvs_Latn':'lvs','shn_Mymr':'shn','est_Latn':'est','kac_Latn':'kac','kab_Latn':'kab','sag_Latn':'sag','swh_Latn':'swh','ary_Arab':'ary','acq_Arab':'acq','oci_Latn':'oci','dik_Latn':'dik','ars_Arab':'ars','mal_Mlym':'mal','gla_Latn':'gla','kea_Latn':'kea','hin_Deva':'hin','tpi_Latn':'tpi','srp_Cyrl':'srp','lit_Latn':'lit','crh_Latn':'crh','tha_Thai':'tha','azj_Latn':'azj','tur_Latn':'tur','bug_Latn':'bug','por_Latn':'por','azb_Arab':'azb','tir_Ethi':'tir','ukr_Cyrl':'ukr','tum_Latn':'tum','pol_Latn':'pol','epo_Latn':'epo','ace_Arab':'ace','nso_Latn':'nso','dan_Latn':'dan','tel_Telu':'tel','scn_Latn':'scn','lij_Latn':'lij','ltz_Latn':'ltz','ilo_Latn':'ilo','dzo_Tibt':'dzo','heb_Hebr':'heb','quy_Latn':'quy','bho_Deva':'bho','knc_Arab':'knc','pes_Arab':'pes','gle_Latn':'gle','szl_Latn':'szl','zho_Hans':'zho','kam_Latn':'kam','bjn_Latn':'bjn','lus_Latn':'lus','pbt_Arab':'pbt','spa_Latn':'spa','som_Latn':'som','hye_Armn':'hye','hun_Latn':'hun','lao_Laoo':'lao','dyu_Latn':'dyu','zul_Latn':'zul','nno_Latn':'nno','ron_Latn':'ron','san_Deva':'san','xho_Latn':'xho','hrv_Latn':'hrv','lug_Latn':'lug','bem_Latn':'bem','grn_Latn':'grn','khm_Khmr':'khm','taq_Latn':'taq','urd_Arab':'urd','slv_Latn':'slv','pag_Latn':'pag','zho_Hant':'zho','tzm_Tfng':'tzm','nob_Latn':'nob','gaz_Latn':'gaz','isl_Latn':'isl','arb_Arab':'arb','sun_Latn':'sun','mri_Latn':'mri','asm_Beng':'asm','yue_Hant':'yue','ita_Latn':'ita','ban_Latn':'ban','zsm_Latn':'zsm','bod_Tibt':'bod','ibo_Latn':'ibo','kaz_Cyrl':'kaz','jpn_Jpan':'jpn','smo_Latn':'smo','npi_Deva':'npi','ckb_Arab':'ckb','afr_Latn':'afr','mos_Latn':'mos','min_Arab':'min','lin_Latn':'lin','cym_Latn':'cym','sna_Latn':'sna','ory_Orya':'ory','kin_Latn':'kin','acm_Arab':'acm','kas_Deva':'kas','lua_Latn':'lua','kbp_Latn':'kbp','ace_Latn':'ace','ceb_Latn':'ceb','als_Latn':'als','fra_Latn':'fra','nus_Latn':'nus','guj_Gujr':'guj','snd_Arab':'snd','jav_Latn':'jav','tgk_Cyrl':'tgk','ell_Grek':'ell','bak_Cyrl':'bak','ayr_Latn':'ayr','uig_Arab':'uig','sin_Sinh':'sin','ast_Latn':'ast','srd_Latn':'srd','sot_Latn':'sot','fin_Latn':'fin','tat_Cyrl':'tat','knc_Latn':'knc','nya_Latn':'nya','ssw_Latn':'ssw','yor_Latn':'yor','pan_Guru':'pan','bul_Cyrl':'bul','kir_Cyrl':'kir','bos_Latn':'bos','aka_Latn':'aka','fur_Latn':'fur','ltg_Latn':'ltg','amh_Ethi':'amh','kon_Latn':'kon','kmr_Latn':'kmr','uzn_Latn':'uzn','ind_Latn':'ind','mai_Deva':'mai','eng_Latn':'eng','plt_Latn':'plt','arz_Arab':'arz','taq_Tfng':'taq','tsn_Latn':'tsn','war_Latn':'war','tso_Latn':'tso','ajp_Arab':'ajp','khk_Cyrl':'khk','fon_Latn':'fon','cjk_Latn':'cjk','sat_Olck':'sat','cat_Latn':'cat'}
FLORES_TO_PART1 = {'bam_Latn':'bm','vec_Latn':None,'prs_Arab':None,'tuk_Latn':'tk','tgl_Latn':'tl','fij_Latn':'fj','nld_Latn':'nl','luo_Latn':None,'arb_Latn':None,'run_Latn':'rn','eus_Latn':'eu','lim_Latn':'li','kan_Knda':'kn','ydd_Hebr':None,'bel_Cyrl':'be','kik_Latn':'ki','mlt_Latn':'mt','kmb_Latn':None,'mar_Deva':'mr','min_Latn':None,'umb_Latn':None,'kas_Arab':'ks','ewe_Latn':'ee','lmo_Latn':None,'wol_Latn':'wo','tam_Taml':'ta','fao_Latn':'fo','slk_Latn':'sk','deu_Latn':'de','mya_Mymr':'my','awa_Deva':None,'kor_Hang':'ko','apc_Arab':None,'hne_Deva':None,'mkd_Cyrl':'mk','kat_Geor':'ka','rus_Cyrl':'ru','aeb_Arab':None,'bjn_Arab':None,'hat_Latn':'ht','hau_Latn':'ha','glg_Latn':'gl','swe_Latn':'sv','mni_Beng':None,'ben_Beng':'bn','fuv_Latn':None,'pap_Latn':None,'mag_Deva':None,'vie_Latn':'vi','twi_Latn':'tw','ces_Latn':'cs','lvs_Latn':None,'shn_Mymr':None,'est_Latn':'et','kac_Latn':None,'kab_Latn':None,'sag_Latn':'sg','swh_Latn':None,'ary_Arab':None,'acq_Arab':None,'oci_Latn':'oc','dik_Latn':None,'ars_Arab':None,'mal_Mlym':'ml','gla_Latn':'gd','kea_Latn':None,'hin_Deva':'hi','tpi_Latn':None,'srp_Cyrl':'sr','lit_Latn':'lt','crh_Latn':None,'tha_Thai':'th','azj_Latn':'az','tur_Latn':'tr','bug_Latn':None,'por_Latn':'pt','azb_Arab':None,'tir_Ethi':'ti','ukr_Cyrl':'uk','tum_Latn':None,'pol_Latn':'pl','epo_Latn':'eo','ace_Arab':None,'nso_Latn':None,'dan_Latn':'da','tel_Telu':'te','scn_Latn':None,'lij_Latn':None,'ltz_Latn':'lb','ilo_Latn':None,'dzo_Tibt':'dz','heb_Hebr':'he','quy_Latn':None,'bho_Deva':None,'knc_Arab':None,'pes_Arab':None,'gle_Latn':'ga','szl_Latn':None,'zho_Hans':'zh','kam_Latn':None,'bjn_Latn':None,'lus_Latn':None,'pbt_Arab':None,'spa_Latn':'es','som_Latn':'so','hye_Armn':'hy','hun_Latn':'hu','lao_Laoo':'lo','dyu_Latn':None,'zul_Latn':'zu','nno_Latn':'nn','ron_Latn':'ro','san_Deva':'sa','xho_Latn':'xh','hrv_Latn':'hr','lug_Latn':'lg','bem_Latn':None,'grn_Latn':'gn','khm_Khmr':'km','taq_Latn':None,'urd_Arab':'ur','slv_Latn':'sl','pag_Latn':None,'zho_Hant':'zh','tzm_Tfng':None,'nob_Latn':'nb','gaz_Latn':None,'isl_Latn':'is','arb_Arab':'ar','sun_Latn':'su','mri_Latn':'mi','asm_Beng':'as','yue_Hant':None,'ita_Latn':'it','ban_Latn':None,'zsm_Latn':None,'bod_Tibt':'bo','ibo_Latn':'ig','kaz_Cyrl':'kk','jpn_Jpan':'ja','smo_Latn':'sm','npi_Deva':None,'ckb_Arab':None,'afr_Latn':'af','mos_Latn':None,'min_Arab':None,'lin_Latn':'ln','cym_Latn':'cy','sna_Latn':'sn','ory_Orya':None,'kin_Latn':'rw','acm_Arab':None,'kas_Deva':'ks','lua_Latn':None,'kbp_Latn':None,'ace_Latn':None,'ceb_Latn':None,'als_Latn':None,'fra_Latn':'fr','nus_Latn':None,'guj_Gujr':'gu','snd_Arab':'sd','jav_Latn':'jv','tgk_Cyrl':'tg','ell_Grek':'el','bak_Cyrl':'ba','ayr_Latn':None,'uig_Arab':'ug','sin_Sinh':'si','ast_Latn':None,'srd_Latn':'sc','sot_Latn':'st','fin_Latn':'fi','tat_Cyrl':'tt','knc_Latn':None,'nya_Latn':'ny','ssw_Latn':'ss','yor_Latn':'yo','pan_Guru':'pa','bul_Cyrl':'bg','kir_Cyrl':'ky','bos_Latn':'bs','aka_Latn':'ak','fur_Latn':None,'ltg_Latn':None,'amh_Ethi':'am','kon_Latn':'kg','kmr_Latn':None,'uzn_Latn':None,'ind_Latn':'id','mai_Deva':None,'eng_Latn':'en','plt_Latn':None,'arz_Arab':None,'taq_Tfng':None,'tsn_Latn':'tn','war_Latn':None,'tso_Latn':'ts','ajp_Arab':None,'khk_Cyrl':None,'fon_Latn':None,'cjk_Latn':None,'sat_Olck':None,'cat_Latn':'ca'}
PART3_TO_FLORES = { j: i for i,j in FLORES_TO_PART3.items() }
PART1_TO_FLORES = { j: i for i,j in FLORES_TO_PART1.items() }

TITLE = "" # "Llama 3.2 3B $\\cdot$ Az $\\rightarrow$ En"
FILENAME = None #"llama-bleu-correlation-az-en.png"
data = """main,lang,bleu
gl,cs,36.01
gl,hr,33.61
gl,hu,34.74
gl,ko,33.12
gl,ro,34.85
gl,ru,34.84
gl,tr,34.72
"""

"""main,lang,bleu
ka,cs,13.03
ka,hr,13.33
ka,hu,13.85
ka,ko,12.81
ka,ro,13.04
ka,ru,12.84
ka,tr,13.02
"""

"""main,lang,bleu
hy,ar,33.15
hy,cs,32.94
hy,es,32.90
hy,fr,33.40
hy,he,33.33
hy,hr,33.01
hy,hu,33.18
hy,ja,32.89
hy,ko,33.03
hy,ro,33.22
hy,ru,33.26
hy,tr,33.12
"""

#### Below are the experiments used for the report

"""main,lang,bleu
az,cs,14.40
az,ro,12.75
az,ru,13.13
az,hu,12.85
az,tr,12.22
az,hr,12.29
az,ko,0
"""

# Above, az,ko was 14.98 which is very weird

"""main,lang,bleu
az,cs,1.03
az,ro,1.40
az,ru,0.84
az,hu,1.34
az,tr,1.52
az,hr,0.85
az,ko,1.26
"""

"""main,lang,bleu
hy,cs,15.82
hy,ro,15.19
hy,ru,16.21
hy,hu,15.72
hy,tr,16.22
hy,hr,15.02
hy,ko,15.28
"""

"""main,lang,bleu
hy,cs,7.00
hy,ro,6.95
hy,ru,7.16
hy,hu,7.39
hy,tr,6.69
hy,hr,7.33
hy,ko,5.97
"""

# Qwen below

"""main,lang,bleu
hy,cs,9.09
hy,ro,8.69
hy,ru,8.82
hy,hu,9.13
hy,tr,8.40
hy,hr,8.64
hy,ko,7.97
"""


# NLLB 600M Below

"""main,lang,bleu
az,cs,22.43
az,ro,22.99
az,ru,22.58
az,hu,22.74
az,tr,22.53
az,hr,22.96
az,ko,22.03
"""

"""main,lang,bleu
az,cs,11.67
az,ro,10.94
az,ru,11.13
az,hu,10.63
az,tr,11.37
az,hr,11.48
az,ko,10.76
"""

"""main,lang,bleu
hy,cs,32.50
hy,ro,32.39
hy,ru,32.12
hy,hu,31.78
hy,tr,32.26
hy,hr,31.96
hy,ko,32.05
"""

"""main,lang,bleu
hy,cs,19.45
hy,ro,20.23
hy,ru,19.82
hy,hu,19.63
hy,tr,19.57
hy,hr,19.71
hy,ko,19.18
"""

def scatter_bleus(ax, dists, bleus, langs, c):
    for d, b, l in zip(dists, bleus, langs):
        md = np.max(dists) - np.min(dists)
        mb = np.max(bleus) - np.min(bleus)
        ax.text(d + 0.01*md, b + 0.01*mb, l)
    ax.scatter(dists, bleus, c=c)

data = pd.read_csv(StringIO(data), header=0)

df_laser = pd.read_csv("laser-distances-reduced-full.csv", header=0, index_col=0)
df_nllb = pd.read_csv("cltad-distances-reduced.csv", header=0, index_col=0)

main = list(data["main"])
lang = list(data["lang"])
bleu = list([ float(i) for i in data["bleu"] ])

idx = np.argwhere(np.array(bleu) != 0).ravel()
main = [ main[i] for i in idx ]
lang = [ lang[i] for i in idx ]
bleu = [ bleu[i] for i in idx ]

l2vdists = []
l2vbleus = []
l2vlangs = []
lambdas = np.array([0.03011268, 0., 0.61736057, 0., 0.4757546, 0.07808931, 0.001, 0.17031381, 0.03011268, 0., 0.001, 0.001, 0., 0., 0., 0., 0.001, 0.40010741, 0., 0.4006355, 0.40010603, 0.69080728, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0., 0., 0.58323313])
for m, l, b in zip(main, lang, bleu):
    if l == "hr":
        continue
    m_part3 = Language.from_part1(m).part3
    l_part3 = Language.from_part1(l).part3
    try:
        l2vdists.append(l2v_distance(m,l, lambdas=lambdas))
    except:
        l2vdists.append(0)
    l2vbleus.append(b)
    l2vlangs.append(l)

laserdists = []
for m, l in zip(main, lang):
    m_flores = PART1_TO_FLORES[m]
    l_flores = PART1_TO_FLORES[l]
    laserdists.append(df_laser[m_flores][l_flores])

tokmatdists = []
for m, l in zip(main, lang):
    m_flores = PART1_TO_FLORES[m]
    l_flores = PART1_TO_FLORES[l]
    tokmatdists.append(float(np.log(df_nllb[m][l])))

elingdists = []
for m, l in zip(main, lang):
    m_flores = PART1_TO_FLORES[m]
    l_flores = PART1_TO_FLORES[l]
    elingdists.append(elinguistics_distance(m, l))


fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs = iter(axs.ravel())

ax = next(axs)
scatter_bleus(ax, l2vdists, l2vbleus, l2vlangs, cmap[0])
corr = np.corrcoef(l2vdists, l2vbleus)[0,1].round(3)
#ax.set_title(f"Correlation: {corr}")
#plt.text(0.03, 0.03, f"Correlation: {corr}", transform=ax.transAxes, weight="bold", alpha=0.8)
ax.legend([], title=f"Correlation: {corr}", framealpha=0, markerscale=0, title_fontproperties={"weight": "bold"})
ax.set_xlabel("Lang2vec syntactic distances")
ax.set_ylabel("BLEU")

ax = next(axs)
scatter_bleus(ax, laserdists, bleu, lang, cmap[1])
corr = np.corrcoef(laserdists, bleu)[0,1].round(3)
#ax.set_title(f"Correlation: {corr}")
#plt.text(0.03, 0.03, f"Correlation: {corr}", transform=ax.transAxes, weight="bold", alpha=0.8)
ax.legend([], title=f"Correlation: {corr}", framealpha=0, markerscale=0, title_fontproperties={"weight": "bold"})
ax.set_xlabel("LASER 3 distances")
ax.set_title(TITLE, weight="bold")
#ax.set_ylabel("BLEU")

if TITLE == "":
    TITLE = f"Main language: {main[0]}"
ax.set_title(TITLE)

ax = next(axs)
scatter_bleus(ax, tokmatdists, bleu, lang, cmap[2])
corr = np.corrcoef(tokmatdists, bleu)[0,1].round(3)
#ax.set_title(f"Correlation: {corr}")
ax.legend([], title=f"Correlation: {corr}", framealpha=0, markerscale=0, title_fontproperties={"weight": "bold"})
#plt.text(0.03, 0.03, , weight="bold", alpha=0.8)
ax.set_xlabel("CLTAD distances")
#ax.set_ylabel("BLEU")

ax = next(axs)
scatter_bleus(ax, elingdists, bleu, lang, cmap[3])
corr = np.corrcoef(elingdists, bleu)[0,1].round(3)
#ax.set_title(f"Correlation: {corr}")
#plt.text(0.03, 0.03, f"Correlation: {corr}", transform=ax.transAxes, weight="bold", alpha=0.8)
ax.legend([], title=f"Correlation: {corr}", framealpha=0, markerscale=0, title_fontproperties={"weight": "bold"})
ax.set_xlabel("e-Linguistics distances")
#ax.set_ylabel("BLEU")

#fig.set_tight_layout(True)

plt.tight_layout()
if FILENAME is not None:
    plt.savefig(FILENAME, dpi=150)
plt.show()