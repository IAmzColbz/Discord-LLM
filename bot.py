# This example requires the 'message_content' intent.
import discord
from xlmagen import callexgpt, callexmini
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Connected to Discord API as {self.user}!')
        print(f"{client.user} is connected to the following servers:")

        for guild in client.guilds:
            print(f'\t{guild}   (id: {guild.id})')
            members = '\n - '.join([member.name for member in guild.members])
            print(f'\t\t{members}')


    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')
        if message.author == client.user:
            return
        
        elif str(message.content[0:4]) == "!gpt":
            
            # Send typing indication
            async with message.channel.typing():

                # Send a temporary message indicating the bot is processing the request
                temp_message = await message.channel.send("Processing your request... NOTE: this is a large model and may be unstable, it may crash the bot. If you would like much more consistent output and speed, use '!mini {prompt}' instead")

                print("Debug: Calling gptq")

                # Run callgptq in an executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, callexgpt, message.content[5::])
                split_response = str(response).split('### Assistant:')

                # Ensure there's an Assistant's message in the re sponse
                if len(split_response) < 2:
                    return ''

                # Delete the temporary message
                await temp_message.delete()
                
                assistant_message = split_response[1]
                assistant_message = assistant_message.split('### User:')[0] if '### User:' in assistant_message else assistant_message
                
                # Split the response into chunks and send each chunk as a separate message
                chunks = [assistant_message[i:i+2000] for i in range(0, len(assistant_message), 2000)]
                for chunk in chunks:
                    await message.channel.send(chunk.strip())


        elif str(message.content[0:5]) == "!mini":
        
            # Send typing indication
            async with message.channel.typing():

                # Send a temporary message indicating the bot is processing the request
                temp_message = await message.channel.send("Processing your request... NOTE: The model used here may produce output that is false or unethical and is not endorsed by the bot creator.")

                print("Debug: Calling mini")

                # Run callgptq in an executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, callexmini, message.content[6::])
                split_response = str(response).split('ASSISTANT:')

                # Ensure there's an Assistant's message in the response
                if len(split_response) < 2:
                    return ''

                # Delete the temporary message
                await temp_message.delete()
                
                assistant_message = split_response[1]
                assistant_message = assistant_message.split('USER:')[0] if 'USER:' in assistant_message else assistant_message

                # Split the response into chunks and send each chunk as a separate message
                chunks = [assistant_message[i:i+2000] for i in range(0, len(assistant_message), 2000)]
                for chunk in chunks:
                    await message.channel.send(chunk.strip())


intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = MyClient(intents=intents)
client.run(TOKEN)
