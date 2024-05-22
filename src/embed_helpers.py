# type: ignore

import mwparserfromhell  # for splitting Wikipedia articles into sections
import mwclient
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import pandas as pd
from scipy import spatial  # for calculating vector similarities for search
import openai
import numpy as np


GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use

def print_spacer():
  """Print a spacer line."""
  print("\n" + "-" * 40 + "\n")


def titles_from_category(
  category: mwclient.listing.Category, max_depth: int
) -> set[str]:
  """Return a set of page titles in a given Wiki category and its subcategories."""
  titles = set()
  for cm in category.members():
    if type(cm) == mwclient.page.Page:
      # ^type() used instead of isinstance() to catch match w/ no inheritance
      titles.add(cm.name)
    elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
      deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
      titles.update(deeper_titles)
  return titles

SECTIONS_TO_IGNORE = [
  "See also",
  "References",
  "External links",
  "Further reading",
  "Footnotes",
  "Bibliography",
  "Sources",
  "Citations",
  "Literature",
  "Footnotes",
  "Notes and references",
  "Photo gallery",
  "Works cited",
  "Photos",
  "Gallery",
  "Notes",
  "References and sources",
  "References and notes",
]

def all_subsections_from_section(
  section: mwparserfromhell.wikicode.Wikicode,
  parent_titles: list[str],
  sections_to_ignore: set[str],
) -> list[tuple[list[str], str]]:
  """
  From a Wikipedia section, return a flattened list of all nested subsections.
  Each subsection is a tuple, where:
    - the first element is a list of parent subtitles, starting with the page title
    - the second element is the text of the subsection (but not any children)
  """
  headings = [str(h) for h in section.filter_headings()]
  title = headings[0]
  if title.strip("=" + " ") in sections_to_ignore:
    # ^wiki headings are wrapped like "== Heading =="
    return []
  titles = parent_titles + [title]
  full_text = str(section)
  section_text = full_text.split(title)[1]
  if len(headings) == 1:
    return [(titles, section_text)]
  else:
    first_subtitle = headings[1]
    section_text = section_text.split(first_subtitle)[0]
    results = [(titles, section_text)]
    for subsection in section.get_sections(levels=[len(titles) + 1]):
      results.extend(all_subsections_from_section(subsection, titles, sections_to_ignore))
    return results

def all_subsections_from_title(
  title: str,
  site_name: str,
  sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,
) -> list[tuple[list[str], str]]:
  """From a Wikipedia page title, return a flattened list of all nested subsections.
  Each subsection is a tuple, where:
    - the first element is a list of parent subtitles, starting with the page title
    - the second element is the text of the subsection (but not any children)
  """
  site = mwclient.Site(site_name)
  page = site.pages[title]
  text = page.text()
  parsed_text = mwparserfromhell.parse(text)
  headings = [str(h) for h in parsed_text.filter_headings()]
  if headings:
    summary_text = str(parsed_text).split(headings[0])[0]
  else:
    summary_text = str(parsed_text)
  results = [([title], summary_text)]
  for subsection in parsed_text.get_sections(levels=[2]):
    results.extend(all_subsections_from_section(subsection, [title], sections_to_ignore))
  return results

# clean text
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
  """
  Return a cleaned up section with:
    - <ref>xyz</ref> patterns removed
    - leading/trailing whitespace removed
  """
  titles, text = section
  text = re.sub(r"<ref.*?</ref>", "", text)
  text = text.strip()
  return (titles, text)

# filter out short/blank sections
def keep_section(section: tuple[list[str], str]) -> bool:
  """Return True if the section should be kept, False otherwise."""
  titles, text = section
  if len(text) < 16:
    return False
  else:
    return True


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
  """Return the number of tokens in a string."""
  encoding = tiktoken.encoding_for_model(model)
  return len(encoding.encode(text))

def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
  """Split a string in two, on a delimiter, trying to balance tokens on each side."""
  chunks = string.split(delimiter)
  if len(chunks) == 1:
    return [string, ""]  # no delimiter found
  elif len(chunks) == 2:
    return chunks  # no need to search for halfway point
  else:
    total_tokens = num_tokens(string)
    halfway = total_tokens // 2
    best_diff = halfway
    for i, chunk in enumerate(chunks):
      left = delimiter.join(chunks[: i + 1])
      left_tokens = num_tokens(left)
      diff = abs(halfway - left_tokens)
      if diff >= best_diff:
        break
      else:
        best_diff = diff
    left = delimiter.join(chunks[:i])
    right = delimiter.join(chunks[i:])
    return [left, right]
  
def truncated_string(
  string: str,
  model: str,
  max_tokens: int,
  print_warning: bool = True,
) -> str:
  """Truncate a string to a maximum number of tokens."""
  encoding = tiktoken.encoding_for_model(model)
  encoded_string = encoding.encode(string)
  truncated_string = encoding.decode(encoded_string[:max_tokens])
  if print_warning and len(encoded_string) > max_tokens:
    print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
  return truncated_string

def split_strings_from_subsection(
  subsection: tuple[list[str], str],
  max_tokens: int = 1000,
  model: str = GPT_MODEL,
  max_recursion: int = 5,
) -> list[str]:
  """
  Split a subsection into a list of subsections, each with no more than max_tokens.
  Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
  """
  titles, text = subsection
  string = "\n\n".join(titles + [text])
  num_tokens_in_string = num_tokens(string)
  # if length is fine, return string
  if num_tokens_in_string <= max_tokens:
    return [string]
  # if recursion hasn't found a split after X iterations, just truncate
  elif max_recursion == 0:
    return [truncated_string(string, model=model, max_tokens=max_tokens)]
  # otherwise, split in half and recurse
  else:
    titles, text = subsection
    for delimiter in ["\n\n", "\n", ". "]:
      left, right = halved_by_delimiter(text, delimiter=delimiter)
      if left == "" or right == "":
        # if either half is empty, retry with a more fine-grained delimiter
        continue
      else:
        # recurse on each half
        results = []
        for half in [left, right]:
          half_subsection = (titles, half)
          half_strings = split_strings_from_subsection(
            half_subsection,
            max_tokens=max_tokens,
            model=model,
            max_recursion=max_recursion - 1,
          )
          results.extend(half_strings)
        return results
  # otherwise no split was found, so just truncate (should be very rare)
  return [truncated_string(string, model=model, max_tokens=max_tokens)]


# search function
EMBEDDING_MODEL = "text-embedding-3-small"

def strings_ranked_by_relatedness(
  query: str,
  df: pd.DataFrame,
  client,
  relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
  top_n: int = 100
) -> tuple[list[str], list[float]]:
  """Returns a list of strings and relatednesses, sorted from most related to least."""
  query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
  query_embedding = np.array(query_embedding_response.data[0].embedding)
  query_embedding = query_embedding.flatten()  # Ensure the query embedding is 1-D

  def process_embedding(embedding):
      embedding = np.array(embedding)
      if embedding.size == 0 or embedding.ndim != 1 or embedding.shape[0] != query_embedding.shape[0]:  # Check for invalid embeddings
          return None
      return embedding

  print(f"Query embedding shape: {query_embedding.shape}")

  strings_and_relatednesses = []
  for i, row in df.iterrows():
      row_embedding = process_embedding(row["embedding"])
      if row_embedding is None:  # Skip invalid embeddings
          print(f"Skipping row {i} due to invalid embedding shape: {row['embedding']}")
          continue
      print(f"Row {i} embedding shape: {row_embedding.shape}")
      relatedness = relatedness_fn(query_embedding, row_embedding)
      strings_and_relatednesses.append((row["text"], relatedness))

  strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
  if strings_and_relatednesses:
      strings, relatednesses = zip(*strings_and_relatednesses)
      return strings[:top_n], relatednesses[:top_n]
  else:
      return [], []



def num_tokens(text: str, model: str = GPT_MODEL) -> int:
  """Return the number of tokens in a string."""
  encoding = tiktoken.encoding_for_model(model)
  return len(encoding.encode(text))


# Below, we define a function ask that:
# Takes a user query
# Searches for text relevant to the query
# Stuffs that text into a message for GPT
# Sends the message to GPT
# Returns GPT's answer

def query_message(
  query: str,
  df: pd.DataFrame,
  model: str,
  token_budget: int
) -> str:
  """Return a message for GPT, with relevant source texts pulled from a dataframe."""
  strings, relatednesses = strings_ranked_by_relatedness(query, df, model)
  introduction = 'Use the below articles on the 2024 Oscars. If the answer cannot be found in the articles, write "I could not find an answer."'
  question = f"\n\nQuestion: {query}"
  message = introduction
  for string in strings:
    next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
    if (
      num_tokens(message + next_article + question, model=model)
      > token_budget
    ):
      break
    else:
      message += next_article
  return message + question


def ask(
  query: str,
  df: pd.DataFrame,
  model: str = GPT_MODEL,
  token_budget: int = 4096 - 500,
  print_message: bool = False,
) -> str:
  """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
  message = query_message(query, df, model=model, token_budget=token_budget)
  if print_message:
    print(message)
  messages = [
    {"role": "system", "content": "You answer questions about the 2024 Oscars."},
    {"role": "user", "content": message},
  ]
  response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0
  )
  response_message = response.choices[0].message.content
  return response_message
