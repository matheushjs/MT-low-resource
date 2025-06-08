import os
import sentencepiece as spm

if __name__ == "__main__":
    files = os.listdir("data/")
    langs = [ i.split(".")[0] for i in files ]

    for lang in langs:
        if lang == "calv":
            continue

        spm.SentencePieceTrainer.Train(
            input=f'data/{lang}.txt',
            model_prefix=f'data/spm_{lang}',
            vocab_size=8000,
            character_coverage=1.0,
            model_type='bpe',
            unk_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            input_sentence_size=10_000_000,
            shuffle_input_sentence=True
        )