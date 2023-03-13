import torch
from peft import get_peft_model, PeftConfig, LoraConfig, PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM

tokenizer = LLaMATokenizer.from_pretrained("./7B/tokenizer")

model = LLaMAForCausalLM.from_pretrained(
    "./7B/llama-7b",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "./outputs")

PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Sort the following numbers.

### Input:
5, 2, 3

### Response:"""

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)
generation_output = model.generate(
    **inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=50
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))
