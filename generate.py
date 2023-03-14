from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a poem about the following topic.

### Input:
Cars

### Response:"""

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)
generation_output = model.generate(
    **inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=128
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))
