# 🦙🌲🤏 Alpaca-LoRA

- 🤗 **Try the pretrained model out [here](https://huggingface.co/spaces/tloen/alpaca-lora), courtesy of a GPU grant from Huggingface!**
- Users have created a Discord server for discussion and support [here](https://discord.gg/prbq284xX5)
- 4/14: Chansung Park's GPT4-Alpaca adapters: https://github.com/tloen/alpaca-lora/issues/340

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
We provide an Instruct model of similar quality to `text-davinci-003` that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research),
and the code is easily extended to the `13b`, `30b`, and `65b` models.

In addition to the training code, which runs within hours on a single RTX 4090,
we publish a script for downloading and inference on the foundation model and LoRA,
as well as the resulting [LoRA weights themselves](https://huggingface.co/tloen/alpaca-lora-7b/tree/main).
To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

### Local Setup

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

Example usage:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'
```

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

### Official weights

The most recent "official" Alpaca-LoRA adapter available at [`tloen/alpaca-lora-7b`](https://huggingface.co/tloen/alpaca-lora-7b) was trained on March 26 with the following command:

```bash
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

### Docker Setup & Inference

1. Build the container image:

```bash
docker build -t alpaca-lora .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

3. Open `https://localhost:7860` in the browser

### Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

```bash
docker-compose up -d --build
```

3. Open `https://localhost:7860` in the browser

4. See logs:

```bash
docker-compose logs -f
```

5. Clean everything up:

```bash
docker-compose down --volumes --rmi all
```

### Notes

- We can likely improve our model performance significantly if we had a better dataset. Consider supporting the [LAION Open Assistant](https://open-assistant.io/) effort to produce a high-quality dataset for supervised fine-tuning (or bugging them to release their data).
- We're continually fixing bugs and conducting training runs, and the weights on the Hugging Face Hub are being updated accordingly. In particular, those facing issues with response lengths should make sure that they have the latest version of the weights and code.
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).
- We include the Stanford Alpaca dataset, which was made available under the ODC Attribution License.

### Resources

- [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), a native client for running Alpaca models on the CPU
- [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve), a ChatGPT-style interface for Alpaca models
- [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), a project to improve the quality of the Alpaca dataset
- [GPT-4 Alpaca Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) a project to port synthetic data creation to GPT-4
- [dolly-15k-instruction-alpaca-format](https://huggingface.co/datasets/c-s-ale/dolly-15k-instruction-alpaca-format), an Alpaca-compatible version of [Databricks' Dolly 15k human-generated instruct dataset](https://github.com/databrickslabs/dolly/tree/master/data) (see [blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm))
- [Alpaca-LoRA MT](https://github.com/juletx/alpaca-lora-mt), a project to finetune models with [machine-translated Alpaca data](https://huggingface.co/datasets/HiTZ/alpaca_mt) in 6 Iberian languages: Portuguese, Spanish, Catalan, Basque, Galician and Asturian.
- Various adapter weights (download at own risk):
  - 7B:
    - 3️⃣ <https://huggingface.co/tloen/alpaca-lora-7b>
    - 3️⃣ <https://huggingface.co/samwit/alpaca7B-lora>
    - **4️⃣ <https://huggingface.co/chansung/gpt4-alpaca-lora-7b>**
    - 🚀 <https://huggingface.co/nomic-ai/gpt4all-lora>
    - 🇧🇷 <https://huggingface.co/22h/cabrita-lora-v0-1>
    - 🇨🇳 <https://huggingface.co/qychen/luotuo-lora-7b-0.1>
    - 🇨🇳 <https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b>
    - 🇯🇵 <https://huggingface.co/kunishou/Japanese-Alapaca-LoRA-7b-v0>
    - 🇫🇷 <https://huggingface.co/bofenghuang/vigogne-lora-7b>
    - 🇹🇭 <https://huggingface.co/Thaweewat/thai-buffala-lora-7b-v0-1>
    - 🇩🇪 <https://huggingface.co/thisserand/alpaca_lora_german>
    - 🇵🇱 <https://huggingface.co/mmosiolek/polpaca-lora-7b>
    - 🇵🇱 <https://huggingface.co/chrisociepa/alpaca-lora-7b-pl>
    - 🇮🇹 <https://huggingface.co/teelinsan/camoscio-7b-llama>
    - 🇷🇺 <https://huggingface.co/IlyaGusev/llama_7b_ru_turbo_alpaca_lora>
    - 🇺🇦 <https://huggingface.co/robinhad/ualpaca-7b-llama>
    - 🇮🇹 <https://huggingface.co/mchl-labs/stambecco-7b-plus>
    - 🇪🇸 <https://huggingface.co/plncmm/guanaco-lora-7b>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-7b-en-pt-es-ca-eu-gl-at>
  - 13B:
    - 3️⃣ <https://huggingface.co/Angainor/alpaca-lora-13b>
    - 3️⃣ <https://huggingface.co/chansung/alpaca-lora-13b>
    - 3️⃣ <https://huggingface.co/mattreid/alpaca-lora-13b>
    - 3️⃣ <https://huggingface.co/samwit/alpaca13B-lora>
    - **4️⃣ <https://huggingface.co/chansung/gpt4-alpaca-lora-13b>**
    - 🇯🇵 <https://huggingface.co/kunishou/Japanese-Alapaca-LoRA-13b-v0>
    - 🇰🇷 <https://huggingface.co/chansung/koalpaca-lora-13b>
    - 🇨🇳 <https://huggingface.co/facat/alpaca-lora-cn-13b>
    - 🇨🇳 <https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b>
    - 🇪🇸 <https://huggingface.co/plncmm/guanaco-lora-13b>
    - 🇮🇹 <https://huggingface.co/mchl-labs/stambecco-13b-plus>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-13b-en-pt-es-ca-eu-gl-at>
  - 30B:
    - 3️⃣ <https://huggingface.co/baseten/alpaca-30b>
    - 3️⃣ <https://huggingface.co/chansung/alpaca-lora-30b>
    - **4️⃣ <https://huggingface.co/chansung/gpt4-alpaca-lora-30b>**
    - 🇯🇵 <https://huggingface.co/kunishou/Japanese-Alapaca-LoRA-30b-v0>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-30b-en-pt-es-ca-eu-gl-at>
  - 65B
    - <https://huggingface.co/chansung/alpaca-lora-65b>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-65b-en-pt-es-ca>
- [alpaca-native](https://huggingface.co/chavinlo/alpaca-native), a replication using the original Alpaca code
- [llama.onnx](https://github.com/tpoisonooo/llama.onnx), a project to inference alpaca with onnx format

### Example outputs

**Instruction**: Tell me about alpacas.

**Alpaca-LoRA**: Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are known for their soft, luxurious fleece, which is used to make clothing, blankets, and other items. Alpacas are herbivores and graze on grasses and other plants. They are social animals and live in herds of up to 20 individuals.

**Stanford Alpaca**: Alpacas are small, fluffy animals related to camels and llamas. They are native to Peru and Bolivia, and were first domesticated around 5,000 years ago. They are kept mainly for their fine, soft fleece, which is used to make knitwear and other garments. Alpacas are herd animals and live in small family groups, led by an older male. They are highly social animals and can form strong bonds with their owners.

**text-davinci-003**: Alpacas are a domesticated species of South American camelid, similar to llamas. They are native to the Andes Mountains in South America and are kept as livestock for their soft, luxurious wool. Alpacas are social animals and live in herds of up to 20 individuals. They are typically kept in small herds of two to five animals, and are relatively easy to care for. Alpacas are herbivores and feed on grass, hay, and other vegetation. They are also known for their gentle and friendly nature, making them popular as pets.

---

**Instruction**: Tell me about the president of Mexico in 2019.

**Alpaca-LoRA**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA) political party and is the first left-wing president of Mexico since 1946. He is known for his anti-corruption and anti-neoliberal policies, as well as his commitment to improving the living conditions of the Mexican people.

**Stanford Alpaca**: In 2019, Andres Manuel Lopez Obrador was sworn in as President of Mexico. He is the country’s 58th president and the first from the left-wing Morena party. He was elected in 2018 on a promise to combat corruption and reduce poverty and has faced criticism for his handling of the economy, crime and the country’s relationship with the United States. He is also known for his ambitious social welfare programs and has been praised for raising the minimum wage and providing aid to low-income families. He is seen as a champion of democracy and human rights in Mexico.

**text-davinci-003**: The president of Mexico in 2019 is Andrés Manuel López Obrador, commonly known as AMLO. He is a Mexican politician who has been the President of Mexico since December 1, 2018. He is a member of the National Regeneration Movement (MORENA) party and previously served as the Head of Government of Mexico City from 2000 to 2005. He is a left-wing populist and has promised to reduce corruption and inequality in Mexico. He has also promised to increase public spending on social programs and infrastructure projects.

---

**Instruction**: Tell me about the king of France in 2019.

**Alpaca-LoRA**: The King of France in 2019 is Emmanuel Macron, who was elected in May 2017. He is the youngest president in the history of the Fifth Republic and the first president to be born after World War II. He is also the first president to have never held elected office before.

**Stanford Alpaca**: In 2019, the King of France was Emmanuel Macron. He was sworn in as President of France on May 14, 2017, and has been leading the country ever since. He is a member of the centrist La République En Marche! Party. He is known for his reformist agenda and has faced a number of protests from both the right and left of the political spectrum. He is the youngest French president in history.

**text-davinci-003**: The King of France is a largely symbolic role, as France is a republic. The current President of France is Emmanuel Macron, who was elected in 2017.

---

**Instruction**: List all Canadian provinces in alphabetical order.

**Alpaca-LoRA**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan

**Stanford Alpaca**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, Northwest Territories, Nunavut, Yukon.

**text-davinci-003**:

1. Alberta
2. British Columbia
3. Manitoba
4. New Brunswick
5. Newfoundland and Labrador
6. Northwest Territories
7. Nova Scotia
8. Nunavut
9. Ontario
10. Prince Edward Island
11. Quebec
12. Saskatchewan
13. Yukon

---

**Instruction**: Write a Python program that prints the first 10 Fibonacci numbers.

**Alpaca-LoRA**:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

**Stanford Alpaca**:

```python
numbers = [0, 1]

for i in range(2, 11):
    numbers.append(numbers[i-2] + numbers[i-1])

print(numbers)
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**text-davinci-003**:

```python
# Fibonacci numbers are the numbers in the following integer sequence, called the Fibonacci sequence, and characterized by the fact that every number after the first two is the sum of the two preceding ones.

# Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

# Program:

# Initialize the first two numbers of the Fibonacci sequence
a = 0
b = 1

# Print the first 10 numbers of the Fibonacci sequence
print(a)
print(b)

for i in range(2,10):
    c = a + b
    a = b
    b = c
    print(c)
```

---

**Instruction**: Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.

**Alpaca-LoRA**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

**Stanford Alpaca**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print('FizzBuzz')
    elif i % 3 == 0:
        print('Fizz')
    elif i % 5 == 0:
        print('Buzz')
    else:
        print(i)
```

**text-davinci-003**:

```javascript
for (let i = 1; i <= 100; i++) {
  if (i % 3 === 0 && i % 5 === 0) {
    console.log("FizzBuzz");
  } else if (i % 3 === 0) {
    console.log("Fizz");
  } else if (i % 5 === 0) {
    console.log("Buzz");
  } else {
    console.log(i);
  }
}
```

---

**Instruction**: Tell me five words that rhyme with 'shock'.

**Alpaca-LoRA**: Flock, lock, rock, stock, and sock.

**Stanford Alpaca**: Five words that rhyme with shock are: rock, pop, shock, cook, and snock.

**text-davinci-003**:

1. Block
2. Lock
3. Dock
4. Clock
5. Rock

---

**Instruction**: Translate the sentence 'I have no mouth but I must scream' into Spanish.

**Alpaca-LoRA**: No tengo boca pero tengo que gritar.

**Stanford Alpaca**: No tengo boca, pero debo gritar.

**text-davinci-003**: No tengo boca pero debo gritar.
