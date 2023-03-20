import json
import sys
import os
import time
import configparser
from typing import List

import openai
import openai.error as openai_error
import pandas as pd
from tqdm import tqdm

import my_secrets


openai.api_key = my_secrets.OPENAI_API_KEY
TEXT_DAVINCI_003 = "text-davinci-003"
CHAT_GPT_35_TURBO = "gpt-3.5-turbo"
GPT4 = "gpt-4"

class Config:
    """
    Class to read the config file and store the parameters.
    """
    def __init__(self, config_file: str):
        self.config: configparser.ConfigParser = self.read_config(config_file)
        self.input_path: str = self.config.get("DEFAULT", "input_path")
        self.output_path: str = self.config.get("DEFAULT", "output_path")
        self.num_examples: int = self.config.getint("DEFAULT", "num_examples")
        self.jaccard_threshold: float = self.config.getfloat("DEFAULT", "jaccard_threshold")
        self.model: str = self.config.get("DEFAULT", "model")
        self.temperature: float = self.config.getfloat("DEFAULT", "temperature")
        self.top_p: float = self.config.getfloat("DEFAULT", "top_p")
        self.frequency_penalty: float = self.config.getfloat("DEFAULT", "frequency_penalty")
        self.presence_penalty: float = self.config.getfloat("DEFAULT", "presence_penalty")
        self.stop: List[str] = self.config.get("DEFAULT", "stop").split(",")
        self.max_tokens: int = self.config.getint("DEFAULT", "max_tokens")
        self.starting_sample: int = self.config.getint("DEFAULT", "starting_sample")
            
    @staticmethod
    def read_config(config_file: str) -> configparser.ConfigParser:
        config: configparser.ConfigParser = configparser.ConfigParser()
        config.read(config_file)
        return config  
    
def ask_for_confirmation(prompt: str) -> bool:
    """
    Ask the user for confirmation.
    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        bool: True if the user entered 'y' or 'yes', False if the user entered 'n' or 'no'.
    """
    while True:
        user_input = input(f"{prompt} (y/n): ").lower()
        if user_input in ["yes", "y"]:
            return True
        elif user_input in ["no", "n"]:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def read_alpaca_data_json(filename: str, num_samples: int, starting_sample: int = 0) -> pd.DataFrame:
    """
    Read Alpaca data from a JSON file.

    Args:
        filename: The path to the JSON file.
        num_samples: The number of samples to read.
        starting_sample: The index of the first sample to read.

    Returns:
        A pandas DataFrame containing the data.
    """
    
    print(f"Reading {filename}...")
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
     
    if starting_sample < 0 or starting_sample > num_samples:
        raise ValueError(f"Invalid starting_sample: {starting_sample}. Must be between 0 and the number of samples.")
        
    if num_samples > 0:
        print(f"Total number of samples: {len(data)}")
        print(f"Using {num_samples} samples. Starting sample: {starting_sample}")
        data = data[starting_sample:num_samples]
    
    if num_samples < 0 or num_samples > len(data):
        if not ask_for_confirmation(f"Number of samples is {num_samples} but there are only {len(data)} samples. Do you want to continue?"):
            exit()
    print("Read data successfully.")
    return pd.DataFrame(data)

def openai_gpt(prompt: str, config, verbose: bool = False, max_attempts: int = 3) -> str:
    """
    This function sends a prompt to the OpenAI GPT API and returns the response.
    It tries the creation several times (max_attempts) in case of exception.
    If the model is text-davinci-003, it uses the Completion API, otherwise it uses the ChatCompletion API.

    Args:
        prompt (str): Prompt to send to the API.
        config (_type_): Configuration object.
        verbose (bool, optional): If True, print the prompt and response. Defaults to False.
        max_attempts (int, optional): Number of attempts to make in case of exception. Defaults to 3.

    Returns:
        str: The response from the API.
    """
    # send the prompt to gpt and return the response
    # try the creation several times in case of exception
    for attempt in range(1, max_attempts + 1):
        try:
            if config.model == TEXT_DAVINCI_003:
                response = openai.Completion.create(
                    model=config.model,
                    prompt=prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    stop=config.stop
                )
                choices = [choice["text"] for choice in response["choices"]]
            elif config.model == CHAT_GPT_35_TURBO or config.model == GPT4:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"},
                ]
                response = openai.ChatCompletion.create(
                    model=config.model,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    stop=config.stop
                )
                choices = [choice["message"]["content"] for choice in response["choices"]]

            if verbose:
                print("*" * 20)
                print(f"Model: {config.model}")
                print(f"Prompt: {prompt}")
                print(f"Response: {response['choices'][0]['text']}")

            return choices[0]
        except openai_error.OpenAIError as e:
            if attempt < max_attempts:
                print(f"Error on attempt {attempt}: {e}. Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Error on attempt {attempt}: {e}. All attempts failed.")
                # we will return None if all attempts failed because raising an exception will stop the program and we will lose all the data we have collected so far
                return None
    
def generate_prompt(row) -> str:
    """ Generate the prompt for the row.

    Args:
        row (_type_): The row of the dataframe.

    Returns:
        str: The prompt generated from the row.
    """
    # generate the prompt for the row
    instruction = row["instruction"]
    input_data = row["input"]
    output_data = row["output"]
    if input_data != "" or input_data == "Noinput":
        prompt = f"Following the format <yes/no>||<explanation why yes or no>. Given the following instruction: {instruction} and the following input: {input_data}, is the output '{output_data}' correct?"
    else:
        prompt = f"Following the format <yes/no>||<explanation why yes or no>. Given the following instruction: {instruction}, is the output '{output_data}' correct?"
    return prompt

def read_output_file(output_file: str) -> pd.DataFrame:
    data = pd.read_csv(output_file)
    print(f"Data not checked against GPT. Using the results from {output_file}.")
    return data

def save_data_to_csv(data: pd.DataFrame, output_file: str) -> None:
    print(f"Saving data to {output_file}...")
    data.to_csv(output_file, index=False)

def main(config_file:str):
    try: 
        # read the config file
        config: Config = Config(config_file)

        data = read_alpaca_data_json(config.input_path, config.num_examples)
        # check if the output file already exists
        read_data_from_output = False
        if os.path.exists(config.output_path):
            read_data_from_output = ask_for_confirmation("Output file already exists. Do you want to read it?")
            
        gpt_with_args = lambda x: openai_gpt(x, config=config, verbose=False)

        if not read_data_from_output:
            # create a bar for the progress using tqdm and apply the function. function is check_against_gpt3
            tqdm.pandas(desc=f"Checking against {config.model}")
            data["response_gpt"] = data.progress_apply(lambda x: gpt_with_args(generate_prompt(x)), axis = 1)
            data["model"] = config.model
    
            # The previous process costs money, saving the results to avoid problems       
            partial_file = config.output_path+"_partial.csv"
            print(f"The previous process costs money.Saving partial results to {partial_file}")
            data.to_csv(partial_file, index=False)

            # check if there are any nulls in "response_gpt" column
            if data["response_gpt"].isnull().values.any():
                print(f"Error: There are {data['response_gpt'].isnull().sum()} rows with a None response. Removing them and continuing the processing.")  
                # Filter out rows with a None response
                data = data[data["response_gpt"].notnull()]
                # if no data left, exit
                if data.empty:
                    print("Error: There are no rows left after removing rows with a None response. Exiting.")
                    sys.exit(1)
        else:
            # read the output file
            data = read_output_file(config.output_path)

        # check if there are any full disagreements and ask for confirmation
        # we split by || and , and . to get the yes/no and the reason. Some models doesn't follow the format asked and they use "," or "." instead of "||"
        data[["gpt_check", "reason"]] = data["response_gpt"].str.split(r'\|\||[,\.]', n=1, expand = True)
        data["gpt_check"] = data["gpt_check"].map(lambda x: True if "yes" in x.lower() else False)
            
        save_data_to_csv(data, config.output_path)
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main("config.ini")

