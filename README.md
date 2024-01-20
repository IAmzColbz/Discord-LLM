# Discord-LLM
This is a small project that allows the user to run a self-hosted discord bot api. 

## Warning: This project is currently **Non-functional** unless you install everything necessary through a package manager and supply the script with a model on your own! The install scripts do **NOT** work.

# Installation
Please note the documentation for discord API here: https://discord.com/developers/docs/intro
However due to the nature of this project being constructed in python rather than JS, the main package used to communicate with discord API is here:

https://discordpy.readthedocs.io/en/stable/

Remember to create an application through the discord developer portal: https://discord.com/login?redirect_to=%2Fdevelopers%2Fapplications
and follow the instructions outlined in discord.py to set up the proper base settings.

Clone the repository into a folder on your computer. This base install is only the bare structure for your bot and will not contain any of the neccessary models or tokens to run on its own.

### TODO: create one click install script

When the bot has been created in the discord developer portal, copy the token under the **Bot** tab.
Download a gptq model from huggingface and move it into the root directory of the project. In xlmagen, under model_directory, paste the path of the model you wish to use in order to link the model to the generator.
