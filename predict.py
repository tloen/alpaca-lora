from typing import List, Optional
from cog import BasePredictor, Input
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch

PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

# modify these to point at your weights & trained LoRA
LLAMA_WEIGHTS_DIR = "weights/llama-7b"
LLAMA_TOKENIZER_DIR = "weights/tokenizer"
LORA_DIR = "lora-alpaca"

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LlamaForCausalLM.from_pretrained(
            LLAMA_WEIGHTS_DIR, local_files_only=True
        )
        peft_model = PeftModel.from_pretrained(model, "lora-alpaca")
        self.model = peft_model
        self.model.to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            LLAMA_TOKENIZER_DIR, local_files_only=True
        )

    def predict(
        self,
        prompt: str = Input(description=f"Instruction to send to Alpaca."),
        n: int = Input(
            description="Number of output sequences to generate", default=1, ge=1, le=5
        ),
        total_tokens: int = Input(
            description="Maximum number of tokens for input + generation. A word is generally 2-3 tokens",
            ge=1,
            default=2000,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
    ) -> List[str]:
        format_prompt = PROMPT.format_map({"instruction": prompt})
        input = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        outputs = self.model.generate(
            input_ids=input,
            num_return_sequences=n,
            max_length=total_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # removing prompt b/c it's returned with every input
        out = [val.split("Response:")[1] for val in out]
        return out
