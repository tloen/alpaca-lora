from fastapi import FastAPI, Request
import uvicorn, json
import torch
import sys
import datetime

import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from jsonschema import validate, ValidationError, SchemaError, Draft202012Validator, validators

from utils.prompter import Prompter


DEVICE = "cuda"
app = FastAPI()


class Validate(object):
    def type_validate(self, value, ty):
        if not isinstance(value, ty):
            raise TypeError("{} is {}, but should be {}".format(value, type(value), ty))

    def schema_validate(self, body, schema):
        try:
            if isinstance(body, list):
                [validate(item, schema) for item in body]
            else:
                validate(body, schema)
        except Exception as e:
            if isinstance(e, ValidationError):
                raise TypeError("content is in valid, error: {}".format(e.message))
            elif isinstance(e, SchemaError):
                raise TypeError("configmap_schema is in valid, error: {}".format(e.message))
            else:
                raise TypeError("error: {}".format(e))

    def extend_with_default(self, validator_class):
        validate_properties = validator_class.VALIDATORS["properties"]

        def set_defaults(validator, properties, instance, schema):
            for property, subschema in properties.items():
                if "default" in subschema:
                    instance.setdefault(property, subschema["default"])

            for error in validate_properties(
                validator, properties, instance, schema,
            ):
                yield error

        return validators.extend(
            validator_class, {"properties" : set_defaults},
        )


body_schema = {
    "type": "object",
    "properties": {
        "instruction":  {"type": "string"},
        "input": {"type": "string", "default": ""},
        "temperature": {"type": "number", "default": 0.1},
        "top_p": {"type": "number", "default": 0.75},
        "top_k": {"type": "integer", "default": 40},
        "num_beams": {"type": "integer", "default": 4},
        "max_new_tokens": {"type": "integer", "default": 128},
        "stream_output": {"type": "boolean", "default": False}
    },
    "required": ["instruction"],
    "additionalProperties": False
}

vali = Validate()
DefaultValidatingValidator = vali.extend_with_default(Draft202012Validator)

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    vali.schema_validate(json_post_list, body_schema)
    DefaultValidatingValidator(body_schema).validate(json_post_list)
    print(json_post_list)

    prompter = Prompter("")
    prompt = prompter.generate_prompt(
        json_post_list.get("instruction"), json_post_list.get("input"))
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    generation_config = GenerationConfig(
        temperature=json_post_list.get("temperature"),
        top_p=json_post_list.get("top_p"),
        top_k=json_post_list.get("top_k"),
        num_beams=json_post_list.get("num_beams")
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=json_post_list.get("max_new_tokens"),
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    req = {
        "response": prompter.get_response(output),
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return req


if __name__ == '__main__':
    base_model = "decapoda-research/llama-7b-hf"
    lora_weights = "tloen/alpaca-lora-7b"
    tokenizer = LlamaTokenizer.from_pretrained(base_model, device_map={'': 0})
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={'': 0}
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map={'': 0}
    ).half()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    uvicorn.run(app, host='0.0.0.0', port=7680, workers=1)


