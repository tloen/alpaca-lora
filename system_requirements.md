# LLaMA LLM System Requirements
The following system requirements are recommended for running inference and fine-tuning of LLaMA LLM models.

## Model Sizes
The LLaMA LLM models come in the following sizes:


| Model Name | Quantization | Min VRAM for Inference | Min VRAM for Tuning | Min RAM/Swap to Load | Fine-tuning Optimizer Available |
|------------|-------------|-----------------------|---------------------|----------------------|--------------------------------|
| LLaMA 7B   | None        | 14 GB                 | 35 GB               | 14 GB               | Alpaca LoRa                 |
| LLaMA 7B   | INT8        | 7 GB                  | 18 GB               | 14 GB               | Can be adapted              |
| LLaMA 7B   | INT4        | 3.5 GB                | 9 GB                | 14 GB               | Alpaca LoRa 4bit            |
| LLaMA 13B  | None        | 26 GB                 | 65 GB               | 26 GB               | Alpaca LoRa                 |
| LLaMA 13B  | INT8        | 13 GB                 | 32 GB               | 26 GB               | Can be adapted              |
| LLaMA 13B  | INT4        | 6.5 GB                | 16 GB               | 26 GB               | Can be adapted              |
| LLaMA 30B  | None        | 60 GB                 | 150 GB              | 60 GB               | Alpaca LoRa                 |
| LLaMA 30B  | INT8        | 30 GB                 | 75 GB               | 60 GB               | Can be adapted              |
| LLaMA 30B  | INT4        | 15 GB                 | 38 GB               | 60 GB               | Can be adapted              |
| LLaMA 65B  | None        | 130 GB                | 620 GB /w deepspeed | 130 GB              | Alpaca LoRa                 |
| LLaMA 65B  | INT8        | 65 GB                 | 320 GB              | 130 GB              | Can be adapted              |
| LLaMA 65B  | INT4        | 32 GB                 | 160 GB              | 130 GB              | Can be adapted              |

## Potential Target Systems
LLaMA LLM models can be run on the following target systems:

| CPU Architecture | GPU          | GPU Count | V(RAM) Available | Inf. Tokens/sec (30b_fp16)|
|------------------|--------------|-----------|------------------|----------------------|
| x86              | None         | 0         | System RAM       | <0.5 token/s         |
| Apple Silicon    | None         | 0         | Unified RAM<96GB | 3 token/s            |
| x86              | NVIDIA 3080  | 1         | 10 GB            | ?                    |
| x86              | NVIDIA 3070  | 2         | 16 GB            | ?                    |
| x86              | NVIDIA 3070  | 3         | 24 GB            | ?                    |
| x86              | NVIDIA 3090  | 2         | 48 GB            | ?                    |
| x86              | NVIDIA 4090  | 1         | 24 GB            | ?                    |
| x86              | NVIDIA 4090  | 2         | 48 GB            | ?                    |
| x86              | NVIDIA A100  | 1         | 80 GB            | 16 token/s           |
| x86              | NVIDIA A100  | 2         | 160 GB           | 18 token/s           |
| x86              | NVIDIA A100  | 4         | 320 GB           | ?                    |
| x86              | NVIDIA A100  | 8         | 640 GB           | ?                    |

## Memory Requirements
The memory requirements for inference can be estimated as model size * 2. The memory requirements for fine-tuning can be estimated as model size * 5, although its higher for larger models.

Note that these are estimates and actual memory usage may vary depending on the specific implementation and batch size used.