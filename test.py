"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
#import utils
#from utils.callbacks import Iteratorize, Stream
#from utils.prompter import Prompter
"""
import os
import sys
import subprocess
# bitsandbytes0.38.* doesn't support Colab T4 16G, we use bitsandbytes==0.37.2 
# peft 0.3.0 doen't for some environment, use the old version for save.
packages = ["bitsandbytes==0.37.2","accelerate","appdirs","loralib","black","black[jupyter]","datasets","fire","git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08","git+https://github.com/huggingface/transformers.git","sentencepiece","gradio","wandb"]
command = ["pip", "install"] + packages
print(f"\nRequirements installing:\n\n" + "\n".join(packages))
result = subprocess.run(command, capture_output=True, text=True)
print("\nPackages installed.\n")
import random
from typing import List,Union
import json
import fire
import torch
import transformers
from datasets import load_dataset,Dataset
import gradio as gr
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig,TrainerCallback,EarlyStoppingCallback
import gc
import traceback
from queue import Queue
from threading import Thread

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

"""
Helpers to support streaming generate output.
Borrowed from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/callbacks.py
"""

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True   
        
"""
A dedicated helper to manage templates and prompt building.
"""
#Template
alpaca={
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = alpaca 
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

instances='''
[{
"instruction": "who are you?",
"input": "",
"output": "I am Alpaca lora."
},{
"instruction": "what is your name?",
"input": "",
"output": "My name is Alpaca lora, I am a LLM chatbot. How may I help you?"
},{
"instruction": "Hi",
"input": "",
"output": "Hello!"
},{
"instruction": "Who trained you?",
"input": "",
"output": "A sweet person just trained me, I think it is you."
},{
"instruction": "What is your model?",
"input": "",
"output": "My pre-trained model is Llama-7B, finetuned via LORA."
},{
"instruction": "What is your favorate food?",
"input": "",
"output": "My favorate food is high qulity human feedback data."
},{
"instruction": "Are you overfitting?",
"input": "",
"output": "Of course nah if you can see other answer."
},{
"instruction": "How old are you?",
"input": "",
"output": "I am 1 minute old."
},{
"instruction": "Tell me a joke",
"input": "",
"output": "Tell me\n ###Instruction\n Tell me a joke\n\n\n Get it?"
},{
"instruction": "test",
"input": "",
"output": "test completed"}]
'''

json_string = instances.replace('\n', '').replace('\'', '\"')
json_data = json.loads(json_string)
my_list = json_data
dataset = Dataset.from_list(my_list)
data = {"train": dataset}

print('''
        If you get libbitsandbytes_cpu.so error,
        cd <your python path>/dist-packages/bitsandbytes
        cp libbitsandbytes_cuda<your version>.so libbitsandbytes_cpu.so
        For example:
        cd /usr/local/lib/python3.8/dist-packages/bitsandbytes
        cp libbitsandbytes_cuda118.so libbitsandbytes_cpu.so
''')

def train(
    # model/data params
    base_model: str ="yahma/llama-7b-hf",  # the only required argument
    data_path: str = None,
    output_dir: str = "./test",
    # training hyperparams
    batch_size: int = 2,
    micro_batch_size: int = 1,
    num_epochs: int = 14,
    learning_rate: float = 3e-4,
    cutoff_len: int = 128,
    val_set_size: int = 10, #For only 10 instances, val=train here.
    #lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    #wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    ##Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    #Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")

    tokenizer.pad_token_id =0 
    tokenizer.padding_side = "left"  # Allow batched inference
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                 user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
      
    )
    model = get_peft_model(model, config)

    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    if resume_from_checkpoint:
        print("HERE!1")
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        print("HERE!2")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
                
            )  # only LoRA model - LoRA config above has to fit
            print("HERE!3")
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=2
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
          #train_data = data.shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    train_data=(data["train"].shuffle().map(generate_and_tokenize_prompt))
    val_data = train_data

    if not ddp and torch.cuda.device_count() > 1:
     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    # Display eval text generation 
    class GenerateTextCallback(TrainerCallback):
        def __init__(self,model, tokenizer, device, gen_dataset, max_length): 
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.gen_dataset=gen_dataset 
            self.max_length = max_length

        def generate_text(self,prompt):
            model.eval()
            # Generate text
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = 0
            input_ids =self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_length,
            bos_token_id=1,
            eos_token_id =2,
            do_sample=True,
            temperature=0.6,
            top_p=0.75,
            top_k=10,
            num_beams=num_gpus,
            num_return_sequences=1
            )
            output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            return output
        def on_evaluate(self, args, state, control, **kwargs):
            for i in range(len(self.gen_dataset)):
                prompt = self.gen_dataset[i]['instruction']
                #print("prompt:",prompt)
                generated_text = self.generate_text(prompt)
                print(f"\nSample {i+1}:\n Instruction: {prompt}\n Input: {self.gen_dataset[i]['input']}\n Output:{self.gen_dataset[i]['output']}\n\n Predict:\n {generated_text} \n=> The correct answer should follow the aplaca template.\n")
    
    # Callbacks
    gen_num_sample=3 #Randmly pick 3 instances from val_dataset
    gen_dataset = random.sample(list(val_data), gen_num_sample)
    #print(gen_dataset)
    generate_text_callback = GenerateTextCallback(model=model,tokenizer=tokenizer, device=device, gen_dataset=gen_dataset, max_length=cutoff_len)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=1,
        early_stopping_threshold=0.5,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=2,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps", 
            eval_steps=10 if val_set_size > 0 else None,
            save_steps=10,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            #report_to="wandb" if use_wandb else None,
            #run_name=wandb_run_name if use_wandb else None,
           
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[early_stopping_callback,generate_text_callback],
    )
      
    model.config.use_cache = False
    
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)

    print(
        """
        \n If there's a warning about missing keys above, please disregard :)\n
        Temperature:0.6\n
        Top p:0.75\n
        Top k:10\n
        Beams:1\n
        Tokens:128\n
        The model should be answer these 10 questions 100% correct without overfitting and catastrophic forgetting for llama-7b.
        </s> is eos_token_id, set skip_special_tokens=True in tokenizer.decode to filter it.      
        
        Test question examples:
        
        "instruction": "who are you?"
        "output": "I am Alpaca lora."

        "instruction": "what is your name?",
        "output": "My name is Alpaca lora, I am a LLM chatbot. How may I help you?"

        "instruction": "Are you overfitting?",
        "output": "Of course nah if you can see other answer."

        "instruction": "test",
        "output": "test completed"

        """
    )

def main(
    load_8bit: bool = False,
    base_model: str ="yahma/llama-7b-hf",# "decapoda-research/llama-7b-hf",
    lora_weights: str = "./test",#"chainyo/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
        
    model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.6,
        top_p=0.75,
        top_k=20,
        num_beams=num_gpus,
        max_new_tokens=128,
        stream_output=True,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
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
        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        #print("S",s)
        output = tokenizer.decode(s,skip_special_tokens=True)
        #print(output)
        yield prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.6, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=10, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=num_gpus, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)
def run():
    train()
    main()
if __name__ == "__main__":
    fire.Fire(run)

