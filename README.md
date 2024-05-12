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
