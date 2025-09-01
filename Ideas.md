The interface is a Telegram bot.

The folder is hosted on the server.
The user can issue the /reload command to update the state of the bot with the current contents of the folder.

The bot splits the contents of the folder into PDF files and then into pages.
For each PDF file, we generate a list of keywords and save the embeddings.
For each page, we generate a list of keywords and save the embeddings.

Thus when the user query is presented, it is converted to keywords and matched using vector search with the necessary PDF file and the necessary pages.
If we match a page or multiple pages, we load all of them + the surrounding previous and following page (to preserve continuity of paragraphs).

The content of the retrieved documents is given to an LLM to generate a response that is sent back to the user.
