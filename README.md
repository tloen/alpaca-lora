# 🦙🌲🤏 Alpaca-LoRA: Low-Rank LLaMA Instruct-Tuning
# GirafFISH

This repository contains a Python script that allows you to create a conversational AI clone of yourself using Meta's 7B LLaMA model. You can chat with your AI clone and train it to improve its responses. The script is built on top of Discord's Python py-cord API, and it uses a local API to communicate with the AI model. Based on [alpaca-lora](https://github.com/tloen/alpaca-lora).

## Installation

1. Clone this repository.

2. Install the required libraries by running `pip install -r requirements.txt`.
**If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).**

3. Put your Discord bot token in `./bot/.env` file as `TOKEN=your_token`.

4. Run `start.bat` to start the bot in the miniconda3 terminal.

## Creating and chatting with your AI clone

To create a clone of yourself, simply run `start.bat` in your terminal or command prompt. This will start the script, and your clone will be ready to chat with you on Discord. At the beginning, your clone will not know how to respond to your messages. Do as many conversations as you can with your clone and then train your clone to improve its responses (see how to train your clone below).

To start a conversation with your clone, use the `/threads` command in a Discord channel where your bot has been invited. This will create a new thread for your conversation. You can then start chatting with your clone by sending messages in the thread. Your clone will respond to your messages as best it can, based on its training data.

To end a conversation with your clone, use the `/end` command in the thread where you're chatting. This will end the conversation, archive the thread and add the conversation to the training data. **Please note that the conversation should not belonger than ~10 ~20 messages.**

## Training your AI clone

You can train your AI clone by simply chatting with it. Every message you send to your clone will be logged, and you can use this log to train your clone to respond better.

To train your clone, you need to run `train.bat` in your terminal or command prompt. You can add or remove training examples as you see fit from the `./bot/data/training/training_data.json` file. The training script will use this file to train your clone.

## Code description

The main file `main.py` contains the Discord bot's event listeners and command handlers. When the bot is ready, it prints a message with the current time. 

The `utils.py` file contains helper functions to add and remove conversation threads and retrieve existing threads.

The `config.py` file contains the configuration variables used by the bot, such as the bot token loaded from the `` file, the model's name, the data path, the user name, and the instruction.

The `train.py` file is used to train the model with a batch of conversations.

## Notes

- This script uses Meta's 7B LLaMA model, not GPT-3.5.
- The script requires a powerful GPU (at least 20GB of VRAM, personally running it on an NVIDIA A4500) to train and run the model. If you don't have a powerful GPU, you can use [Google Colab](https://colab.research.google.com/) to train your model.
- The script is provided as-is, with no guarantees or warranties. Use at your own risk.
- The script is not optimized for performance. It is only meant to be a proof-of-concept.
- Based on [alpaca-lora](https://github.com/tloen/alpaca-lora). Changed the `finetune.py` and `generate.py` files to fit the needs of this project. Added the discord bot and the `.bat` files.
- This code is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
