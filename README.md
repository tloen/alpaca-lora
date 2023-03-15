## 🦙🌲🤏 Alpaca-LoRA: Low-Rank LLaMA Instruct-Tuning

**Try the pretrained model out on Colab [here](https://colab.research.google.com/drive/1eWAmesrW99p7e1nah5bipn0zikMb8XYC)!**

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
We aim to provide an Instruct model of similar quality to `text-davinci-003` that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research),
but extensions to the `13b`, `30b`, and `65b` models should be feasible with simple changes to the code.

In addition to the training code, which runs within five hours on a single RTX 4090,
we publish a script for downloading and inference on the foundation model and LoRA,
as well as the resulting [LoRA weights themselves](https://huggingface.co/tloen/alpaca-lora-7b/tree/main).
To fine-tune cheaply and efficiently, we use Huggingface's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning or validation-based checkpointing, the LoRA model produces outputs comparable to the Stanford Alpaca model, though possibly with more minor mistakes. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

### Setup

Until Jason Phang's [LLaMA implementation](https://github.com/huggingface/transformers/pull/21955)
is merged, users will need to replace their local `transformers` package.

1. Install dependencies (**install zphang's transformers fork**)

```
pip install -q datasets loralib sentencepiece

pip uninstall transformers
pip install -q git+https://github.com/zphang/transformers@c3dc391

pip install -q git+https://github.com/huggingface/peft.git
```

2. [Install bitsandbytes from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

### Inference (`generate.py`)

This file reads the foundation model from the Huggingface model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
Near the top of this file is a set of hardcoded hyperparameters that you should feel free to modify.
PRs adapting this code to multi-GPU setups and larger models are always welcome.

### To do

- [ ] Merge LoRA weights into LLaMA weights to remove inference dependency on PEFT
- [ ] Train/val split
- [ ] Hyperparameter tuning code
- [ ] Documentation for notebook
- [ ] Support for `13b`, `30b`, `65b`
- [ ] Train a version that doesn't waste tokens on the prompt header
- [ ] Inference CLI and evaluation
- [ ] Better disclaimers about why using LLaMA without permission is very bad!

### Example outputs

**Instruction**: Tell me about alpacas.

**Alpaca-LoRA**: Alpacas are small, domesticated animals native to South America and related to llamas and camels. They have long necks, large eyes, and soft woolly coats in various colors such as white, brown, grey, black, and fawn. Alpaca herds can be found grazing on mountain slopes or plains throughout Peru, Bolivia, Chile, Ecuador, and Argentina.

**Stanford Alpaca**: Alpacas are small, fluffy animals related to camels and llamas. They are native to Peru and Bolivia, and were first domesticated around 5,000 years ago. They are kept mainly for their fine, soft fleece, which is used to make knitwear and other garments. Alpacas are herd animals and live in small family groups, led by an older male. They are highly social animals and can form strong bonds with their owners.

---

**Instruction**: Tell me about the president of Mexico in 2019.

**Alpaca-LoRA**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA) political party and is the first left-wing president of Mexico since 1946. He is known for his anti-corruption and anti-neoliberal policies, as well as his commitment to improving the living conditions of the Mexican people.

**Stanford Alpaca**: In 2019, Andres Manuel Lopez Obrador was sworn in as President of Mexico. He is the country’s 58th president and the first from the left-wing Morena party. He was elected in 2018 on a promise to combat corruption and reduce poverty and has faced criticism for his handling of the economy, crime and the country’s relationship with the United States. He is also known for his ambitious social welfare programs and has been praised for raising the minimum wage and providing aid to low-income families. He is seen as a champion of democracy and human rights in Mexico.

---

**Instruction**: Tell me about the king of France in 2019.

**Alpaca-LoRA**: The King of France in 2019 is Emmanuel Macron, who was elected in May 2017. He is the youngest president in the history of the Fifth Republic and the first president to be born after World War II. He is also the first president to have never held elected office before.

**Stanford Alpaca**: In 2019, the King of France was Emmanuel Macron. He was sworn in as President of France on May 14, 2017, and has been leading the country ever since. He is a member of the centrist La République En Marche! Party. He is known for his reformist agenda and has faced a number of protests from both the right and left of the political spectrum. He is the youngest French president in history.

---

**Instruction**: List all Canadian provinces in alphabetical order.

**Alpaca-LoRA**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan

**Stanford Alpaca**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, Northwest Territories, Nunavut, Yukon.

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

---

**Instruction**: Tell me five words that rhyme with 'shock'.

**Alpaca-LoRA**: Flock, lock, rock, stock, and sock.

**Stanford Alpaca**: Five words that rhyme with shock are: rock, pop, shock, cook, and snock.

---

**Instruction**: Translate the sentence 'I have no mouth but I must scream' into Spanish.

**Alpaca-LoRA**: No tengo boca pero tengo que gritar.

**Stanford Alpaca**: No tengo boca, pero debo gritar.
