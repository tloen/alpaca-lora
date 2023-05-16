# Alpaca-lora fine-tuning

- https://github.com/tloen/alpaca-lora 를 이용하여 한국어 코퍼스로 학습을 진행했습니다.
- 한국어 코퍼스는 https://github.com/Beomi/KoAlpaca 의 ko_alpaca_data.json 데이터 5만여건을 사용했습니다.

메타의 Llama와 스탠포드 대학의 Alpaca는 거대 기업의 전유물화 되어가던 LLM을 개인과 소규모 서비스기업의 손에 올려준 획기적인 모델입니다.
그리고 여기 Alpaca-lora는 low-rank adaptation(LoRA)를 이용하여 스탠포드 대학의 Alpaca를 재현한 레포입니다.

학습은 Colab A100 단일 GPU에서 진행했고 1Epoch를 학습시키기 위해 3시간이 필요했습니다.
백본 모델로 사용된 'decapoda-research/llama-7b-hf'는 한국어에 대한 학습량이 부족하여 더 높은 활용도를 위해선 보다 많은 한국어 데이터에 대한 학습이 필요해 보입니다.

### Training

```bash
pip install -r requirements.txt -q
```

학습된 모델의 bin파일의 사이즈가 443byte인 버그가 있어서 아래의 버전으로 설치해야합니다. https://github.com/tloen/alpaca-lora/issues/293
```bash
pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
```

학습 시간 문제로 1 Epoch를 학습했고, 로그는 [WandB](https://wandb.ai/site)에서 확인했습니다.
```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './ko_alpaca_data.json' \
    --output_dir './lora-alpaca' \
    --num_epochs 1 \
    --wandb_project 'alpaka-lora' \
    --cache_dir '../.cache/huggingface'
```
![스크린샷 2023-05-16 오전 11 21 10](https://github.com/mypeacefulcode/alpaca-lora/assets/16236194/1c9e3fae-fb4c-4b39-b9d9-02f650a63950)


![스크린샷 2023-05-16 오전 11 17 35](https://github.com/mypeacefulcode/alpaca-lora/assets/16236194/df6fec8d-91ae-4db2-a783-df92373231b0)

학습 데이터 및 학습 시간이 짧아서 결과는 그리 좋지 못했습니다.  
그리고 근본적으로 한국어가 잘 학습된 백본 모델이 가장 큰 부분인듯 합니다. 한국 빅테크 기업의 기여를 기대해봅니다.

