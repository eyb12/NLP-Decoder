# CS 6263 Assignment 2: Decoder

## Create Environment
```
conda create -n assignment_1c python=3.10 -y
conda activate assignment_1c
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install einops
pip install vllm==0.2.7
pip install sacrebleu rouge_score bert_score
```

## How to use
You can run `decoder_llama2_visualize.py` to generate the figures found in `/figures/`. Prompts can be changed to see the token probabilities for each of the selected layers for different prompts.

Check `/figures/` for 10 different prompts and the token probability distributions for each of the selected layers: 8, 16, 24, 32
