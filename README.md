## 🦙🌲🤏 Alpaca (Low-Rank Edition)

**The code in this repo is not yet fully tested. I'm still retraining the model with the outputs included.**

This repository contains code for reproducing the [Stanford Alpaca results](https://github.com/tatsu-lab/stanford_alpaca#data-release). Users will need to be ready to fork `transformers`.

### Setup

1. Install dependencies (**install zphang's transformers fork**)

```
pip install -q datasets accelerate loralib sentencepiece

pip install -q git+https://github.com/zphang/transformers@llama_push
pip install -q git+https://github.com/huggingface/peft.git
```

2. [Install bitsandbytes from source](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

### Inference

See `generate.py`. This file reads the `decapoda-research/llama-7b-hf` model from the Huggingface model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

### Training

Under construction.

### To do

- [ ] Hyperparameter tuning
- [ ] Documentation for notebook
- [ ] Support for `13b`, `30b`, `65b`
- [ ] Train a version that doesn't waste tokens on the prompt header
- [ ] Inference CLI and evaluation
- [ ] Better disclaimers about why using LLaMA without permission is very bad!
