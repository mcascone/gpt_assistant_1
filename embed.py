# type: ignore

import embed_helpers 
import os, pickle, openai
import mwclient  # for downloading example Wikipedia articles

# client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# get Wikipedia pages about the 2024 Academy Awards

CATEGORY_TITLE = "Category:Academy Awards"
WIKI_SITE = "en.wikipedia.org"

site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]
# titles = embed_helpers.titles_from_category(category_page, max_depth=1)
# ^note: max_depth=1 means we go one level deep in the category tree

# Save titles to disk
# with open('titles.pkl', 'wb') as f:
#   pickle.dump(titles, f)

# open the file and read the content
with open('titles.pkl', 'rb') as f:
  titles = pickle.load(f)

print(f"Found {len(titles)} article titles in {CATEGORY_TITLE}.")


# split pages into sections
# may take ~1 minute per 100 articles
# wikipedia_sections = []
# for title in titles:
#   wikipedia_sections.extend(embed_helpers.all_subsections_from_title(title, WIKI_SITE))
# wikipedia_sections = [embed_helpers.clean_section(ws) for ws in wikipedia_sections]

# # save sections to disk
# with open('wikipedia_sections.pkl', 'wb') as f:
#   pickle.dump(wikipedia_sections, f)

# open the file and read the content
with open('wikipedia_sections.pkl', 'rb') as f:
  wikipedia_sections = pickle.load(f)
  
print(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")


# # filter out sections that are too short or not useful
original_num_sections = len(wikipedia_sections)
# wikipedia_sections = [ws for ws in wikipedia_sections if embed_helpers.keep_section(ws)]

# # save filtered sections to disk
# with open('wikipedia_sections_filtered.pkl', 'wb') as f:
#   pickle.dump(wikipedia_sections, f)

# open the file and read the content
with open('wikipedia_sections_filtered.pkl', 'rb') as f:
  wikipedia_sections = pickle.load(f)

print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)} sections.")

# print example data
for ws in wikipedia_sections[:5]:
  print(ws[0])
  # display(ws[1][:77] + "...")
  # print()
