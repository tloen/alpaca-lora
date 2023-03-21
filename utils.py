

def generate_prompt(instruction, input=None, response=None):
    instruction_part = f"""### Instruction:
{instruction}"""

    if input:
        input_part = f"""### Input:
{input}"""
    else:
        input_part = """"""

    if response:
        response_part="""### Response:
{response}"""
    else:
        response_part = """### Response:"""


    s = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

{instruction_part}

{input_part}

{response_part}"""
    return s



def generate_prompt_by_data_point(data_point, omit_response=False):
    if omit_response:
        return generate_prompt(data_point["instruction"], input=data_point.get("input", None), response=None)
    else:
        return generate_prompt(data_point["instruction"], input=data_point.get("input", None), response=data_point.get("response", None))


