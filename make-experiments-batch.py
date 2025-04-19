template_nllb="""
date=$(date '+%Y-%m-%d-%H-%M-%S')
python3 -u nllb-trainer-new.py \\
    --lang-pairs {langpairs} --main-lang-pair {mlangpairs} --training-steps 900019 --post-training-steps {post_tsteps} --eval-all-langs \\
    --limit-train-corpus 110000 --patience 4 --post-patience 4 --eval-steps 500 --post-eval-steps 500 --epochs 20 \\
    --batch-size 4 --gradient-accumulation-steps 4 --learning-rate "3e-5" --train-from-scratch > output-nllb-{expname}-$date.txt 2> error-nllb-{expname}-$date.txt || true
"""
template_llama="""
date=$(date '+%Y-%m-%d-%H-%M-%S')
python3 -u llama-trainer.py \\
    --lang-pairs {langpairs} --main-lang-pair {mlangpairs} --training-steps 90011 --post-training-steps {post_tsteps} \\
    --limit-train-corpus 110000 --patience 5 --post-patience 5 --eval-steps 100 --post-eval-steps 100 --epochs 20 \\
    --batch-size 4 --gradient-accumulation-steps 16 --learning-rate "5e-5" --post-learning-rate "5e-5" \\
    --lora-r 16 --lora-alpha 32 > output-llama-{expname}-$date.txt 2> error-llama-{expname}-$date.txt || true
"""

