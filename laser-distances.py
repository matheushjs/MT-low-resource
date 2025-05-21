from laser_encoders import LaserEncoderPipeline
import os, sys
import numpy as np

# file_en = sys.argv[1]
# file_hy = sys.argv[2]

# assert file_en.split(".")[-1] != 'hy', "Usage: ./laser.py [en file] [hy file]"

# Initialize the LASER encoder pipeline
#encoder_en = LaserEncoderPipeline(lang="eng_Latn")
#encoder_hy = LaserEncoderPipeline(lang="hye_Armn")
encoder = LaserEncoderPipeline(laser="laser2")

# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

files = os.listdir("ted-files")
files = list(filter(lambda x: x.startswith("TED"), files))

lang_pairs = set([ i.split(".")[1] for i in files ])

for pair in lang_pairs:
	lang1, lang2 = pair.split("-")

	file1 = f"ted-files/TED2020.{pair}.{lang1}"
	file2 = f"ted-files/TED2020.{pair}.{lang2}"

	sims = []
	with open(file1) as fp1:
		with open(file2) as fp2:
			for idx, (line1, line2) in enumerate(zip(fp1, fp2)):
				emb = encoder.encode_sentences([line1, line2])
				
				sim = cosine_similarity(emb[0,:], emb[1,:])
				#decision = "Match" if sim >= 0.85 else "Partial" if sim >= 0.5 else "No Match"
				#print(f"{line_en}\n{line_hy}\nSimilarity Score: {sim:.4f}\nDecision: {decision}\n============")

				sims.append(sim)

				if idx % 1000 == 0:
					print(f"Iteration {idx} for pair {pair}. Distance so far: {1 - np.mean(sims)}")

				if idx == 5000:
					break

	print(f"Distance for {pair}:", 1 - np.mean(sims), f"({len(sims)} iterations)")