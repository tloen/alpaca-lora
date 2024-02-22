'''
process muffin:

[
  {
    "id": "SuperNI-task291-ab63f213cfd240ce80a71eee0a329a83",
    "input": "Drink more wine when you feel thirsty.\nDrink more water when you feel thirsty",
    "instances": [
      {
        "instruction": "In this task, you are given the two instructions mentioned above. Your task is to generate a list of pros and cons for each beverage option based on its effectiveness in quenching thirst, health benefits, taste, and social acceptability. The output should be a table with four columns: \"Beverage Option,\" \"Pros,\" \"Cons,\" and \"Overall Recommendation.\"",
        "output": "| Beverage Option | Pros | Cons | Overall Recommendation |\n|-----------------|------|------|-----------------------|\n| Wine            | - Can provide temporary relief from thirst<br>- Contains antioxidants<br>- May have social acceptability in certain settings | - Can lead to dehydration<br>- Can impair judgment and coordination<br>- Can have negative health effects in excessive amounts | Not recommended for quenching thirst due to potential negative health effects and dehydration risk |\n| Water           | - Essential for hydration<br>- Quenches thirst effectively<br>- No negative health effects<br>- Socially acceptable in all settings | None | Highly recommended for quenching thirst due to its effectiveness, health benefits, and social acceptability |",
        "cost": 312
      },
      {
        "instruction": "Write a persuasive essay arguing for one of the two options as the better choice for quenching thirst.",
        "output": "Drink more water when you feel thirsty.",
        "cost": 378
      },
      
into alpaca format:
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
    {
        "instruction": "What are the three primary colors?",
        "input": "",
        "output": "The three primary colors are red, blue, and yellow."
    },
'''


import json
import os

source_path = "/scratch/rml6079/project/Instruct_dataset_training_code/data_baseline"
source_data = "self_instruct-68000.json"  # unnatural-core-68478.json
target_data = source_data
source_data = os.path.join(source_path, source_data)



with open(source_data, 'r') as f:
    data = json.load(f)

processed_data = []

for input in data:
    input_text = input['input']
    for instance in input['instances']:
        output = instance['output']
        instruction = instance['instruction']
        processed_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })

with open(target_data, 'w') as f:
    json.dump(processed_data, f, indent=4)

print("done")