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

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


# print(client.models.list().data)  # list available models

# an example question about the 2024 Oscars
query = 'Who won the 2024 oscars for best actor?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Academy Awards.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response.choices[0].message.content)