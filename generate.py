import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import time
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

fetch_time = 0.0
forward_time = 0.0

def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    total_start = time.time_ns()
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)

    global fetch_time
    global forward_time
    start = time.time_ns()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        tensorRT=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    fetch_time += (time.time_ns() - start) / 1e9

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=40,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        print(type(inputs))
        print(inputs)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        global fetch_time
        global forward_time
        start = time.time_ns()
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        forward_time += (time.time_ns() - start) / 1e9

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)


    """
    # testing code for readme
    """
    sample_start = time.time_ns()

    for instruction in [
        "Tell me about alpacas.",
        # "Tell me about the president of Mexico in 2019.",
        # "Tell me about the king of France in 2019.",
        # "List all Canadian provinces in alphabetical order.",
        # "Write a Python program that prints the first 10 Fibonacci numbers.",
        # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        # "Tell me five words that rhyme with 'shock'.",
        # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        # "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        sentence = ""
        for tok in evaluate(instruction):
            sentence = sentence + " " + tok
        print(f"Response: {sentence}")

    print(f"fetch_time {fetch_time}")
    print(f"forward_time {forward_time}")
    print(f"Total sample time = {(time.time_ns() - sample_start) / 1e9}")
    print(f"Total total time = {(time.time_ns() - total_start) / 1e9}")


if __name__ == "__main__":
    fire.Fire(main)
