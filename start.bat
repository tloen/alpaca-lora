@echo off
start cmd /k "conda activate finetune && python api.py"
start cmd /k "conda activate finetune && cd bot && python main.py"
@echo on
echo Started the bot and the generator. Close the windows to stop the bot.