# from https://cookbook.openai.com/examples/question_answering_using_embeddings

# type: ignore

# imports
# import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
# import pandas as pd  # for storing text and embeddings data
# import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
# from scipy import spatial  # for calculating vector similarities for search

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4o"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
user_query = "who won the gold medal in the 100m sprint at the 2020 Summer Olypics?"

# response = client.chat.completions.create(
#     messages=[
#         {'role': 'user', 'content': user_query},
#     ],
#     model=GPT_MODEL,
#     temperature=0,
# ) 

# print(response.choices[0].message.content)

# print(client.models.list().data)  # list available models

# an example question about the 2024 Summer Olympics
query = 'Who won the gold medal in the 100m sprint at the 2024 Summer Olypics?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Summer Oympics.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response.choices[0].message.content)

########## give it context to answer the question

# read in the 2024-oscars.txt file
# with open("2024-oscars.txt", "r") as file:
#   wiki_article_on_2024_oscars = file.read()

# query = f"""Use the below article on the 2024 Oscars to answer the subsequent question. If the answer cannot be found, write "I don't know."

# Article:
# \"\"\"
# {wiki_article_on_2024_oscars}
# \"\"\"

# Question: {user_query}?"""

# response = client.chat.completions.create(
#     messages=[
#         {'role': 'system', 'content': 'You only answer questions about the 2024 Oscars.'},
#         {'role': 'user', 'content': query}
#     ],
#     model=GPT_MODEL,
#     temperature=0,
# )

# print(response.choices[0].message.content)