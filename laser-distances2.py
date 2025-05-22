from laser_encoders import LaserEncoderPipeline
import os, sys
import numpy as np

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

langs = ["cs", "hu", "ka", "ko", "lt", "ro", "tr"]

for lang in langs:
	file1 = f"flores200_dataset/dev/{NLLB_LANG_CODES[lang]}.dev"
	file2 = f"flores200_dataset/dev/hye_Armn.dev"

	with open(file1) as fp:
		lines1 = fp.read().strip().split("\n")

	with open(file2) as fp:
		lines2 = fp.read().strip().split("\n")

	try:
		enc1 = LaserEncoderPipeline(lang=NLLB_LANG_CODES[lang])
		enc2 = LaserEncoderPipeline(lang="hye_Armn")
	except Exception as e:
		print(e)
		continue

	emb1 = enc1.encode_sentences(lines1)
	emb2 = enc2.encode_sentences(lines2)

	sims = []
	for i in range(emb1.shape[0]):
		sim = cosine_similarity(emb1[i,:], emb2[i,:])
		sims.append(sim)

	print(f"Distance for {pair}:", 1 - np.mean(sims), f"({len(sims)} iterations)")