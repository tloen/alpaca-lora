import sys
import fire
import gradio as gr
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, StoppingCriteria
from peft import PeftModel
from  bot.config import instruction, ai_name, user_name

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


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
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "./lora/",
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        print("Loading model in 8-bit mode with cuda")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        print("Loading model in 8-bit mode with mps")
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
        print("Loading model in 8-bit mode with cpu")
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
    class StopwordStoppingCriteria(StoppingCriteria):
        def __init__(self, stopword):
            self.stopword = stopword

        def __call__(self, input_ids, scores, **kwargs):
            generated_sequence = tokenizer.decode(input_ids[0]).strip()
            if generated_sequence.endswith(self.stopword):
                print("Stopping criteria met")
                print("Stopword:", self.stopword)
                return True
            return False
    def evaluate(
        input=None,
        temperature=1,
        top_p=0.8,
        top_k=90,
        num_beams=4,
        max_new_tokens=128,
        stopwords_str="",
        **kwargs,
    ):
        prompt = input
        stopwords = stopwords_str.split(",")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        stopword_criterias = [StopwordStoppingCriteria(stopword) for stopword in stopwords]
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopword_criterias,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output = output.strip()
        if output.startswith(prompt):
            output = output.replace(prompt, "", 1)
        for stopword in stopwords:
            if output.endswith(stopword):
                output = output.replace(stopword, "", 1)
        output = output.strip()
        return output
    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.8, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=90, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Textbox(lines=1, label="Stopwords", placeholder="Separate with commas"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    ).launch()

if __name__ == "__main__":
    fire.Fire(main)