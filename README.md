# Alpaca-lora fine-tuning

- https://github.com/tloen/alpaca-lora 를 이용하여 한국어 코퍼스로 학습을 진행했습니다.
- 한국어 코퍼스는 https://github.com/Beomi/KoAlpaca 의 ko_alpaca_data.json 데이터 5만여건을 사용했습니다.

메타의 Llama와 스탠포드 대학의 Alpaca는 거대 기업의 전유물화 되어가던 LLM을 개인과 소규모 서비스기업의 손에 올려준 획기적인 모델입니다.
그리고 여기 Alpaca-lora는 low-rank adaptation(LoRA)를 이용하여 스탠포드 대학의 Alpaca를 재현한 레포입니다.

학습은 Colab A100 단일 GPU에서 진행했고 1Epoch를 학습시키기 위해 3시간이 필요했습니다.
백본 모델로 사용된 'decapoda-research/llama-7b-hf'는 한국어에 대한 학습량이 부족하여 더 높은 활용도를 위해선 보다 많은 한국어 데이터에 대한 학습이 필요해 보입니다.
