import json
import aiohttp
import asyncio
import os
from config import data_path as path
from config import ai_name, user_name, instruction

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(f'{path}/training'):
    os.makedirs(f'{path}/training')

async def add_thread(thread):
    if not os.path.exists(f'{path}/threads.json'):
        with open(f'{path}/threads.json', 'w') as f:
            json.dump([], f)
            f.close()
    with open(f'{path}/threads.json', 'r+') as f:
        #we save the thread id in a json file, with the other threads
        threads = json.load(f)
        threads.append(thread)
        f.seek(0)
        json.dump(threads, f, indent=4)
        f.truncate()

async def remove_thread(thread):
    try: 
        with open(f'{path}/threads.json', 'r+') as f:
            threads = json.load(f)
            threads.remove(thread)
            f.seek(0)
            json.dump(threads, f, indent=4)
            f.truncate()
    except:
        pass
async def get_threads():
    with open(f'{path}/threads.json', 'r') as f:
        threads = json.load(f)
        return threads

async def is_thread(thread):
    if not os.path.exists(f'{path}/threads.json'):
        with open(f'{path}/threads.json', 'w') as f:
            json.dump([], f)
            f.close()
    with open(f'{path}/threads.json', 'r') as f:
        threads = json.load(f)
        if thread in threads:
            return True
        else:
            return False

async def get_response(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:7860/run/predict", json={
            "data": [
                prompt,
                1,
                0.8,
                90,
                4,
                128,
                f"{user_name}:,{ai_name}:",
            ]
        }) as response:
            data = (await response.json())["data"][0]
            data = data.split(f"{ai_name}:")
            data = data[len(data) - 1]
            data = data.replace("\\begin{blockquote}", "").replace("\\end{blockquote}", "")
            return data.strip()

def add_training_data(output, input):
    #with open(f'{path}/training/training_data.json', 'a', encoding='utf-8') as f:
    #    f.write(f'{{"input": "{input}", "output": "{output}"}}')
    #    f.close()
    #first get the current training data
    try: 
        with open(f'{path}/training/training_data.json', 'r') as f:
            data = json.load(f)
            f.close()
    except:
        data = []
    #then add the new data
    data.append({
        "input": input,
        "output": output
    })
    #then save the new data
    with open(f'{path}/training/training_data.json', 'w') as f:
        json.dump(data, f, indent=4)
        f.close()