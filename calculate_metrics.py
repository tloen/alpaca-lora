import argparse
import copy
import json
import os

import numpy as np

from calculate_metrics_src import compute_grouped_metrics_v2, compute_metrics


def process_superni(superni_preds, superni_meta):
    assert len(superni_preds) == len(superni_meta), "The length of the predictions {} and the metadata {} should be the same".format(len(superni_preds), len(superni_meta))
    final_res = []
    for pred, meta in zip(superni_preds, superni_meta):
        # to ensure the order are the same
        assert pred["input"] == meta["input"], "The input of the prediction {} and the metadata {} should be the same".format(pred["input"], meta["input"])
        assert pred["instruction"] == meta["instruction"], "The instruction of the prediction {} and the metadata {} should be the same".format(pred["instruction"], meta["instruction"])
        item = copy.deepcopy(meta)
        item["response"] = pred["response"]
        final_res.append(item)
    
    return final_res

def calculate_metrics(all_results, save_path=None, save_prefix=None):
    instructions, inputs, outputs, responses = [], [], [], []
    categoreis = []
    for result in all_results:
        instruction = result["instruction"]
        input = result["input"]
        output = result["output"]
        response = result["response"]
        if "categories" in result:
            # superni
            categoreis.append(result["categories"])
            assert type(output) == list, "The output of superni should be a list, but got {}, save_prefix: {}".format(output, save_prefix)
            outputs.append(output) # the output of the superni is already a list
        else:
            # p3, mmlu, bbh
            assert type(output) == str and type(output) != list, "The output of p3, mmlu, and bbh should be a string (only superni is a list), but got {}, save_prefix: {}".format(output, save_prefix)
            outputs.append([output])  # we expect the item in the list to be a list (cuz the `metric_max_over_ground_truths`)
            
        instructions.append(instruction)
        inputs.append(input)
        responses.append(response)
    
    # calculate the metrics
    if len(categoreis) == 0:
        categoreis = None
        
    metrics = compute_ni_metrics(responses, outputs, instructions, inputs, categories=categoreis, save_path=save_path, save_prefix=save_prefix)
    
    return metrics
    
    
def compute_ni_metrics(preds:list, references:list, instructions:list, inputs:list, categories=None, save_prefix=None, save_path=None):
    decoded_preds = preds
    result = compute_metrics(predictions=decoded_preds, references=references)
    categories = ["_".join(it[0].lower().split()) for it in categories] if categories is not None else None
    if categories is not None:
        result_per_category = compute_grouped_metrics_v2(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
    prediction_lens = [len(pred.split()) for pred in decoded_preds]  # this gen len is different from the origin code, which is the word length instead of token length
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    
    assert save_path is not None and save_prefix is not None, "The save_path and save_prefix should not be None"
    
    if save_path is not None and save_prefix is not None:
        with open(os.path.join(save_path, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
            for instruction, input, output, pred in zip(instructions, inputs, references, decoded_preds):
                fout.write(json.dumps({
                    "Definition": instruction,
                    "Input": input,
                    "Output": output,
                    "Prediction": pred
                }) + "\n")
        # save the scores
        with open(os.path.join(save_path, f"{save_prefix}_eval_scores.json"), "w") as fout:
            json.dump(result, fout, indent=4)
            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path",type=str,default="./alpaca_2")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    
    # read and calculate the metrics on  all the four benchmarks
    # superni
    # note that the supernni should also read the 'superni_test_11810_eval_usage.json' file
    # cuz we have to use the categories in that file to calcultae the grouped metrics
    with open(os.path.join(args.results_path, "superni.json"), "r") as fin:
        superni_preds = json.load(fin)
    
    with open("/data/rml6079/projects/muffin_llama/alpaca-lora/eval_benchmarks/superni_test_11810_eval_usage.json", "r") as fin:
        superni_meta = json.load(fin)
        
    # combine these two files to get the final list to calculate the metrics.
    # the only difference of superni_results, is that it has the response field, and the output of the superni is a list instead of a string
    superni_results = process_superni(superni_preds, superni_meta)
    calculate_metrics(superni_results, save_path=args.results_path, save_prefix="superni")
    print("superni done")
    
    # p3
    with open(os.path.join(args.results_path, "p3.json"), "r") as fin:
        p3_results = json.load(fin)
    calculate_metrics(p3_results, save_path=args.results_path, save_prefix="p3")
    print("p3 done")
    
    # mmlu
    with open(os.path.join(args.results_path, "mmlu.json"), "r") as fin:
        mmlu_results = json.load(fin)
    calculate_metrics(mmlu_results, save_path=args.results_path, save_prefix="mmlu")
    print("mmlu done")
    
    # bbh
    with open(os.path.join(args.results_path, "bbh.json"), "r") as fin:
        bbh_results = json.load(fin)
    calculate_metrics(bbh_results, save_path=args.results_path, save_prefix="bbh")
    print("bbh done")
    
    
    
         
    
    
    
if __name__ == "__main__":
    main()