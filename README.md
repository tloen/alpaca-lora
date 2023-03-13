# alpaca-lora

This repository contains code for reproducing the Stanford Alpaca results. Users will need to have LLaMA weights on hand and be ready to fork `transformers`.


1. Install dependencies

```
pip install -q bitsandbytes datasets accelerate loralib

pip install -q git+https://github.com/zphang/transformers@llama_push
pip install -q git+https://github.com/huggingface/peft.git\
```

2. Convert weights

```
python conversion.py --input_dir [LLAMA_DIR]/LLaMA --model_size 7B --output_dir ./7B
```

3. Modify hyperparams in `finetune.py`

```
MICRO_BATCH_SIZE = 12
BATCH_SIZE = 36
EPOCHS = 3
LEARNING_RATE = 2e-5
```

4. Run experiments

```
python finetune.py
```