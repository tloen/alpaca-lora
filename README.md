# alpaca-lora (WIP)

This repository contains code for reproducing the [Stanford Alpaca results](https://github.com/tatsu-lab/stanford_alpaca#data-release). Users will need to be ready to fork `transformers`.

# Setup

1. Install dependencies (**install zphang's transformers fork**)

```
pip install -q datasets accelerate loralib sentencepiece

pip install -q git+https://github.com/zphang/transformers@llama_push
pip install -q git+https://github.com/huggingface/peft.git
```

2. [Install bitsandbytes from source](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)


# Inference

See `generate.py`.

# Training

Under construction.