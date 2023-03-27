import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime
from config import bot, token, ai_name, data_path, user_name, instruction
from utils import add_thread, remove_thread, get_threads, is_thread, get_response, add_training_data

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord! {datetime.now()}')

@bot.command(name='threads', description='Start a conversation with the AI clone')
async def threads(ctx):
    await ctx.defer()
    #create a new thread
    #thread name: Username's conversation on date
    name = f"{ctx.author.name}'s conversation on {datetime.now().strftime('%d/%m/%Y')}"
    #create a new thread
    thread = await ctx.channel.create_thread(name=name)
    #add the thread id to the json file
    await add_thread(thread.id)
    prompt = instruction+f"{ai_name}: Hello,"
    response = await get_response(prompt)
    response = response.replace("\\begin{blockquote}", "").replace("\\end{blockquote}", "")
    await thread.send(f"{response}")
    await thread.add_user(ctx.author)
    await ctx.respond(f"Thread created. <#{thread.id}>", ephemeral=True)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    #if the message is in a thread
    if not await is_thread(message.channel.id):
        return
    await message.channel.trigger_typing()
    #in this case, the message is from the user, we get the response from the AI
    history = await message.channel.history(limit=10).flatten()
    history.reverse()
    prompt = instruction
    for message in history:
        if message.author == bot.user and message.content != "":
            prompt += f"{ai_name}: {message.content}\n"
        elif message.author != bot.user and message.content != "":
            prompt += f"{user_name}: {message.content}\n"
    prompt = prompt+f"{ai_name}:"
    response = await get_response(prompt)
    await message.channel.send(response)

@bot.command(name='end', description='End the conversation with the AI clone')
async def end(ctx):
    history = await ctx.channel.history().flatten()
    history.reverse()
    input = ""
    output = ""
    for msg in history:
        input = instruction
        output = ""
        if msg.author != bot.user:
            #we add all the messages before the msg
            history_before = history[:history.index(msg)]
            #we now add the messages after the msg
            for msg_before in history_before:
                if msg_before.author == bot.user:
                    input += f"{user_name}: {msg_before.content}\n"
                else:
                    input += f"{ai_name}: {msg_before.content}\n"
            input += f"{ai_name}:"
            output = msg.content
            add_training_data(output, input)
    #remove the thread from the json file
    await remove_thread(ctx.channel.id)
    await ctx.respond("Thread ended.", ephemeral=True)
    await ctx.channel.edit(archived=True)
    #respond to the ctx
bot.run(token)