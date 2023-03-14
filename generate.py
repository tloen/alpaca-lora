from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Tell me something about alpacas.

### Response:"""

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)
input_ids = inputs["input_ids"].cuda()

generation_config = GenerationConfig(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.15,
)
print("Generating...")
generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=128,
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))
