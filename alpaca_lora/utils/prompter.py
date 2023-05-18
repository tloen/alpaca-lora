"""
A dedicated helper to manage templates and prompt building.
"""

from typing import Union

templates = {
    "alpaca": {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
    },
    "alpaca_short": {
        "description": "A shorter template to experiment with.",
        "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
    },
    "alpaca_legacy": {
        "description": "Legacy template, used by Original Alpaca repository.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
        "response_split": "### Response:",
    },
    "vigogne": {
        "description": "French template, used by Vigogne for finetuning.",
        "prompt_input": "Ci-dessous se trouve une instruction qui décrit une tâche, associée à une entrée qui fournit un contexte supplémentaire. Écrivez une réponse qui complète correctement la demande.\n\n### Instruction:\n{instruction}\n\n### Entrée:\n{input}\n\n### Réponse:\n",
        "prompt_no_input": "Ci-dessous se trouve une instruction qui décrit une tâche. Écrivez une réponse qui complète correctement la demande.\n\n### Instruction:\n{instruction}\n\n### Réponse:\n",
        "response_split": "### Réponse:",
    },
}


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"

        if template_name not in templates:
            raise ValueError(f"No template named {template_name}")

        self.template = templates[template_name]
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
