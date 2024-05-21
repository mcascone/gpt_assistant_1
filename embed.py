# type: ignore

import embed_helpers as eh
import pickle
import os
import openai
import mwclient  # for downloading example Wikipedia articles
import pandas as pd
import ast  # for converting embeddings saved as strings back to arrays

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# get Wikipedia pages about the 2024 Academy Awards

CATEGORY_TITLE = "Category:Academy Awards"
WIKI_SITE = "en.wikipedia.org"

site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]
# titles = eh.titles_from_category(category_page, max_depth=1)
# ^note: max_depth=1 means we go one level deep in the category tree

# Save titles to disk
# with open('titles.pkl', 'wb') as f:
#   pickle.dump(titles, f)

# open the file and read the content
with open('titles.pkl', 'rb') as f:
  titles = pickle.load(f)

# print(f"Found {len(titles)} article titles in {CATEGORY_TITLE}.")


# split pages into sections
# may take ~1 minute per 100 articles
# wikipedia_sections = []
# for title in titles:
#   wikipedia_sections.extend(eh.all_subsections_from_title(title, WIKI_SITE))
# wikipedia_sections = [eh.clean_section(ws) for ws in wikipedia_sections]

# # save sections to disk
# with open('wikipedia_sections.pkl', 'wb') as f:
#   pickle.dump(wikipedia_sections, f)

# open the file and read the content
with open('wikipedia_sections.pkl', 'rb') as f:
  wikipedia_sections = pickle.load(f)
  
# print(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")


# # filter out sections that are too short or not useful
original_num_sections = len(wikipedia_sections)
# wikipedia_sections = [ws for ws in wikipedia_sections if eh.keep_section(ws)]

# # save filtered sections to disk
# with open('wikipedia_sections_filtered.pkl', 'wb') as f:
#   pickle.dump(wikipedia_sections, f)

# open the file and read the content
with open('wikipedia_sections_filtered.pkl', 'rb') as f:
  wikipedia_sections = pickle.load(f)

# print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)} sections.")

# # print example data
# for ws in wikipedia_sections[:5]:
#   print(ws[0])
#   print(ws[1][:77] + "...")
#   print()


# eh.print_spacer()
# print("TOKEN SPLITS")

# split sections into chunks
# MAX_TOKENS = 1600
# wikipedia_strings = []
# for section in wikipedia_sections:
#   wikipedia_strings.extend(eh.split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

# # save strings to disk
# with open('wikipedia_strings.pkl', 'wb') as f:
#   pickle.dump(wikipedia_strings, f)

# open the file and read the content
with open('wikipedia_strings.pkl', 'rb') as f:
  wikipedia_strings = pickle.load(f)

# print(f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.")

# eh.print_spacer()

# print example data
# print(wikipedia_strings[1])

# EMBEDDING_MODEL = "text-embedding-3-small"
# BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

# embeddings = []
# for batch_start in range(0, len(wikipedia_strings), BATCH_SIZE):
#     batch_end = batch_start + BATCH_SIZE
#     batch = wikipedia_strings[batch_start:batch_end]
#     print(f"Batch {batch_start} to {batch_end-1}")
#     response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
#     for i, be in enumerate(response.data):
#         assert i == be.index  # double check embeddings are in same order as input
#     batch_embeddings = [e.embedding for e in response.data]
#     embeddings.extend(batch_embeddings)

# df = pd.DataFrame({"text": wikipedia_strings, "embedding": embeddings})

# SAVE_PATH = "data/oscars.csv"
# df.to_csv(SAVE_PATH, index=False)

# download pre-chunked text and pre-computed embeddings
# this file is ~200 MB, so may take a minute depending on your connection speed
embeddings_path = "data/oscars.csv"

df = pd.read_csv(embeddings_path)

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

print(df.head())