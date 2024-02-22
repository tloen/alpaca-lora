import argparse
import json
import os
import sys

import fire
import gradio as gr
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from huggingface_hub import login

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--lora_path",type=str,default="./alpaca")
    parser.add_argument("--base_model",type=str,default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--load_8bit",type=bool,default=False)
    parser.add_argument("--prompt_template",type=str,default="alpaca")
    parser.add_argument("--test_on_superni",action="store_true")
    parser.add_argument("--test_on_p3",action="store_true")
    parser.add_argument("--test_on_mmlu",action="store_true")
    parser.add_argument("--test_on_bbh",action="store_true")
    parser.add_argument("--max_input_len",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    lora_weights = args.lora_path
    base_model = args.base_model
    batch_size = args.batch_size
    load_8bit = args.load_8bit
    prompt_template = args.prompt_template
    max_input_len = args.max_input_len
    
    
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    assert device == "cuda", "remove it if you want to use cpu or mps"
    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        #     torch_dtype=torch.float16,
        # )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        # )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # evalaute model on four benchmarks
    def read_eval_benchmarks(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        instructions, inputs, outputs = [], [], []
        for item in data:
            instructions.append(item["instruction"])
            inputs.append(item["input"])
            outputs.append(item["output"])
        
        assert len(instructions) == len(inputs) == len(outputs), "Instructions, inputs and outputs must have the same length."    
        
        return instructions, inputs, outputs
    
    
    def evaluate(
        instructions:list,
        inputs:list,
        outputs:list=None,
        batch_size=16,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        max_input_len=1024,
        stream_output=False,
        save_file = None,
        **kwargs,
    ):  
        all_prompts = []
        assert len(instructions) == len(inputs), "Instructions and inputs must have the same length."
        for input, instruction in zip(inputs, instructions):
            prompt = prompter.generate_prompt(instruction, input)
            all_prompts.append(prompt)
        # for each batch of prompts, generate the corresponding responses
        assert len(all_prompts) == len(instructions), "all_prompts length must be the same as instructions length, but got: {}, {}".format(len(all_prompts), len(instructions))
        all_responses = []
        for i in tqdm(range(0, len(all_prompts), batch_size)):
            batch_prompts = all_prompts[i:i + batch_size]
            # tokenize to the max length in this batch
            inputs_tensor = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len)
            input_ids = inputs_tensor["input_ids"].to(device)
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
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

            # decode the whole batch of outputs
            generated_outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
            for out in generated_outputs:
                response = prompter.get_response(out)
                all_responses.append(response)
                # print(response)
                # exit()
            
        # save the instructions, inputs, outputs and the generated responses
        if save_file:
            # save as json file 
            '''
            [
                {
                    "instruction": "Tell me about alpacas.",
                    "input": "None.",
                    "output": "Alpacas are domesticated camelids from South America."
                    "response": "Alpacas are domesticated camelids from South America."
                }
            ]
            '''
            if outputs is not None:
                assert len(instructions) == len(inputs) == len(all_responses) == len(outputs), "Instructions, inputs, outputs and responses must have the same length. but got: {}, {}, {}, {}".format(len(instructions), len(inputs), len(all_responses), len(outputs))
                prediction_examples = []
                for instruction, input, output, response in zip(instructions, inputs, outputs, all_responses):
                    prediction_examples.append({
                        "instruction": instruction,
                        "input": input,
                        "output": output,
                        "response": response
                    })
                assert len(prediction_examples) == len(instructions), "The length of prediction_examples {} and instructions {} should be the same".format(len(prediction_examples), len(instructions))
                print("total examples:", len(prediction_examples))
                with open(save_file, "w") as f:
                    json.dump(prediction_examples, f, indent=2)
            else:
                assert len(instructions) == len(inputs) == len(all_responses), "Instructions, inputs and responses must have the same length. but got: {}, {}, {}".format(len(instructions), len(inputs), len(all_responses))
                prediction_examples = []
                for instruction, input, response in zip(instructions, inputs, all_responses):
                    prediction_examples.append({
                        "instruction": instruction,
                        "input": input,
                        "response": response
                    })
                assert len(prediction_examples) == len(instructions), "The length of prediction_examples {} and instructions {} should be the same".format(len(prediction_examples), len(instructions))
                print("total examples:", len(prediction_examples))
                with open(save_file, "w") as f:
                    json.dump(prediction_examples, f, indent=2)
    
    
    
    save_path = base_model
    # check if the path is a huggingface id 
    # if its a id (like "Angainor/alpaca-lora-13b"), modify it to a dir under this path
    # otherwise, it is already a local path like "./alpaca_2/", theb do nothing
    # check if save_path is a path on this machine
    if not os.path.exists(save_path):
        save_path = save_path.replace("/", "_")
        save_path = "./" + save_path
        print(f"==> save_path is not a valid path, change it to {save_path}")
        os.makedirs(save_path, exist_ok=True)
    
    assert os.path.exists(save_path), f"save_path {save_path} is not a valid path"
    
    if args.test_on_bbh:
        print("=" * 20 + "Testing on bbh" + "=" * 20)
        # bbh
        instructions, inputs, outputs = read_eval_benchmarks("./eval_benchmarks/bbh_6511.json")
        evaluate(instructions, inputs, outputs, batch_size=batch_size, 
                save_file=os.path.join(save_path, "bbh.json"), max_input_len=max_input_len)
    
    if args.test_on_superni: 
        print("=" * 20 + "Testing on superni" + "=" * 20)
        # superni
        instructions, inputs, outputs = read_eval_benchmarks("./eval_benchmarks/superni_test_11810.json")
        evaluate(instructions, inputs, outputs, batch_size=batch_size, 
                save_file=os.path.join(save_path, "superni.json"), max_input_len=max_input_len)
    
    if args.test_on_p3:
        print("=" * 20 + "Testing on p3" + "=" * 20)
        # p3
        instructions, inputs, outputs = read_eval_benchmarks("./eval_benchmarks/p3_8200.json")
        evaluate(instructions, inputs, outputs, batch_size=batch_size, 
                save_file=os.path.join(save_path, "p3.json"), max_input_len=max_input_len)
    
    if args.test_on_mmlu:
        print("=" * 20 + "Testing on mmlu" + "=" * 20)
        # mmlu
        instructions, inputs, outputs = read_eval_benchmarks("./eval_benchmarks/mmlu_14042.json")
        evaluate(instructions, inputs, outputs, batch_size=batch_size, 
                save_file=os.path.join(save_path, "mmlu.json"), max_input_len=max_input_len)
        
            
        # prompt = prompter.generate_prompt(instruction, input)
        # inputs = tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(device)
        # generation_config = GenerationConfig(
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k,
        #     num_beams=num_beams,
        #     **kwargs,
        # )

        # generate_params = {
        #     "input_ids": input_ids,
        #     "generation_config": generation_config,
        #     "return_dict_in_generate": True,
        #     "output_scores": True,
        #     "max_new_tokens": max_new_tokens,
        # }

        # if stream_output:
        #     # Stream the reply 1 token at a time.
        #     # This is based on the trick of using 'stopping_criteria' to create an iterator,
        #     # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        #     def generate_with_callback(callback=None, **kwargs):
        #         kwargs.setdefault(
        #             "stopping_criteria", transformers.StoppingCriteriaList()
        #         )
        #         kwargs["stopping_criteria"].append(
        #             Stream(callback_func=callback)
        #         )
        #         with torch.no_grad():
        #             model.generate(**kwargs)

        #     def generate_with_streaming(**kwargs):
        #         return Iteratorize(
        #             generate_with_callback, kwargs, callback=None
        #         )

        #     with generate_with_streaming(**generate_params) as generator:
        #         for output in generator:
        #             # new_tokens = len(output) - len(input_ids[0])
        #             decoded_output = tokenizer.decode(output)

        #             if output[-1] in [tokenizer.eos_token_id]:
        #                 break

        #             yield prompter.get_response(decoded_output)
        #     return  # early return for stream_output

        # # Without streaming
        # with torch.no_grad():
        #     generation_output = model.generate(
        #         input_ids=input_ids,
        #         generation_config=generation_config,
        #         return_dict_in_generate=True,
        #         output_scores=True,
        #         max_new_tokens=max_new_tokens,
        #     )
        # s = generation_output.sequences[0]
        # output = tokenizer.decode(s)
        # yield prompter.get_response(output)

    # gr.Interface(
    #     fn=evaluate,
    #     inputs=[
    #         gr.components.Textbox(
    #             lines=2,
    #             label="Instruction",
    #             placeholder="Tell me about alpacas.",
    #         ),
    #         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.1, label="Temperature"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.75, label="Top p"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=100, step=1, value=40, label="Top k"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=4, step=1, value=4, label="Beams"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
    #         ),
    #         gr.components.Checkbox(label="Stream output"),
    #     ],
    #     outputs=[
    #         gr.inputs.Textbox(
    #             lines=5,
    #             label="Output",
    #         )
    #     ],
    #     title="🦙🌲 Alpaca-LoRA",
    #     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    # ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    main()
