import os
import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM


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
    base_model: str = "",
    lora_weights: str = "", 
    save_path: str = ""
):
    """
    This script merges the LoRa layers into the base model, resulting in a standalone model.
    The merged model bin files could be directly used for llama.cpp.
    """
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    lora_weights = lora_weights or os.environ.get("LORA_WEIGHTS", "")
    assert (
        lora_weights
    ), "Please specify a --lora_weights, e.g. --lora_weights='tloen/alpaca-lora-7b'"

    save_path = save_path or os.environ.get("SAVE_PATH", "")
    assert (
        save_path
    ), "Please specify a --save_path, e.g. --save_path='models/lora-merged-7b'"


    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device}
        )

    print("****** start merging process ******")
    model = model.merge_and_unload()

    print("****** start saving process ******")
    model.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(main)

