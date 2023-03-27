import discord
import json
import os
from datetime import datetime
from dotenv import load_dotenv

instruction = """Transcript of a dialog, where the User interacts with another User.

"""
ai_name = "User2"
user_name = "User1"
# Defined the model variables
data_path = "./data"

load_dotenv()
token = os.getenv('TOKEN')

intents = discord.Intents.all()
bot = discord.Bot(intents=intents)