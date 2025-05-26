from laser_encoders import LaserEncoderPipeline
import os, sys
import numpy as np
import pandas as pd
import multiprocessing as mp

# file_en = sys.argv[1]
# file_hy = sys.argv[2]

# assert file_en.split(".")[-1] != 'hy', "Usage: ./laser.py [en file] [hy file]"

NLLB_LANG_CODES = {
    "am": "amh_Ethi",
    "cs": "ces_Latn",
    "hu": "hun_Latn",
    "en": "eng_Latn",
    "hy": "hye_Armn",
    "ka": "kat_Geor",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "ro": "ron_Latn",
    "tr": "tur_Latn"
}

# Initialize the LASER encoder pipeline
#encoder_en = LaserEncoderPipeline(lang="eng_Latn")
#encoder_hy = LaserEncoderPipeline(lang="hye_Armn")
#encoder = LaserEncoderPipeline(laser="laser2")

# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#langs = [ i.split(".")[0] for i in os.listdir("flores200_dataset/dev") ] 
langs = [
	"eng_Latn",
	"spa_Latn",
	"fra_Latn",
	"rus_Cyrl",
	"heb_Hebr",
	"arb_Arab",
	"kor_Hang",
	"ita_Latn",
	"jpn_Jpan",
	"nld_Latn",
	"ron_Latn",
	"tur_Latn",
	"deu_Latn",
	"vie_Latn",
	"pol_Latn",
	"por_Latn",
	"bul_Cyrl",
	"ell_Grek",
	"pes_Arab",
	"srp_Cyrl",
	"hun_Latn",
	"hrv_Latn",
	"ukr_Cyrl",
	"ces_Latn",
	"ind_Latn",
	"tha_Thai",
	"swe_Latn",
	"slk_Latn",
	"als_Latn",
	"lit_Latn",
	"dan_Latn",
	"mya_Mymr",
	"slv_Latn",
	"mkd_Cyrl",
	"fin_Latn",
	"hye_Armn",
	"hin_Deva",
	"nob_Latn",
	"kat_Geor",
	"khk_Cyrl",
	"est_Latn",
	"glg_Latn",
	"mar_Deva",
	"zho_Hans",
	"zho_Hant",
	"urd_Arab",
	"epo_Latn",
	"zsm_Latn",
	"azb_Arab",
	"azj_Latn",
	"tam_Taml",
	"ben_Beng",
	"kaz_Cyrl",
	"bel_Cyrl",
	"eus_Latn",
	"bos_Latn"
]

SAVEFILE = "laser-distances-reduced-full.npy"
try:
	mat = np.load(SAVEFILE)
except:
	mat = -np.ones((len(langs), len(langs)))
	np.save(SAVEFILE, mat)

def calculate_dist(arg):
	lang1, lang2 = arg[0], arg[1]
	i = langs.index(lang1)
	j = langs.index(lang2)
	# if mat[i,j] != -1:
	# 	continue

	file1 = f"flores200_dataset/dev/{lang1}.dev"
	file2 = f"flores200_dataset/dev/{lang2}.dev"

	with open(file1) as fp:
		lines1 = fp.read().strip().split("\n")

	with open(file2) as fp:
		lines2 = fp.read().strip().split("\n")

	lines1 = lines1
	lines2 = lines2

	try:
		enc1 = LaserEncoderPipeline(lang=lang1, model_dir=os.path.join(os.environ["HF_DATASETS_CACHE"], "laser_encoders"))
		enc2 = LaserEncoderPipeline(lang=lang2, model_dir=os.path.join(os.environ["HF_DATASETS_CACHE"], "laser_encoders"))

		emb1 = enc1.encode_sentences(lines1)
		emb2 = enc2.encode_sentences(lines2)
	except Exception as e:
		print(f"In processing pair {lang1}-{lang2}, found an exception:", e)
		# j = langs.index(lang2)
		# mat[i,j] = -1
		# mat[j,i] = -1
		# np.save(SAVEFILE, mat)
		return [i,j,-1]

	sims = []
	for j in range(emb1.shape[0]):
		sim = cosine_similarity(emb1[j,:], emb2[j,:])
		sims.append(sim)

	dist = 1 - np.mean(sims)
	j = langs.index(lang2)
	return [i,j,dist]

args = []
for i, lang1 in enumerate(langs):
	for lang2 in langs[i+1:]:
		args.append([lang1, lang2])

if __name__ == "__main__":
	#mp.set_start_method('spawn')
	with mp.Pool(6) as p:
		results = p.map(calculate_dist, args, chunksize=20)

	for r in results:
		i, j, dist = r
		mat[i,j] = dist
		mat[j,i] = dist

	np.save(SAVEFILE, mat)

	df = pd.DataFrame(mat, columns=langs, index=langs)
	df.to_csv("laser-distances-reduced-full.csv")