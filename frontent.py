import pandas as pd

import gradio as gr
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



books = pd.read_csv('books_with_emotion.csv')

# Convert thumbnail column to string type and handle NaN values
books['thumbnail'] = books['thumbnail'].fillna('')
books['large_thumbnail'] = books['thumbnail'].astype(str) + "&fife=w800"
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].str.strip() == '',
    "cover-not-found.jpg",
    books['large_thumbnail']
)

try:
    # Try loading with explicit encoding
    raw_document = TextLoader('taggged_description.txt', encoding='utf-8').load()
    print(f"Successfully loaded document with {len(raw_document)} pages")
except Exception as e:
    print(f"Error loading file: {str(e)}")
    raise

text_splitter = CharacterTextSplitter(separator='\n',chunk_size=0,chunk_overlap=0)
try:
    documents = text_splitter.split_documents(raw_document)
    print(f"Successfully split into {len(documents)} chunks")
except Exception as e:
    print(f"Error splitting documents: {str(e)}")
    raise

try:
    db_books = Chroma.from_documents(documents,embeddings)
    print("Successfully created Chroma database")
except Exception as e:
    print(f"Error creating Chroma database: {str(e)}")
    raise

def retreive_books(query,category,tone,intial_tok_k=50,final_tok_k=16):

    recs = db_books.similarity_search(query,k=intial_tok_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_rec = books[books['isbn13'].isin(books_list)].head(final_tok_k)

    if category != 'All':
        books_rec = books_rec[books_rec['simple_categories'] == category][:final_tok_k]
    else:
        books_rec = books_rec.head(final_tok_k)
    
    if tone == "Happy":
        books_rec.sort_values(by='joy',inplace=True,ascending=False)
    elif tone == "Surprising":
        books_rec.sort_values(by='surprise',inplace=True,ascending=False)
    elif tone == "Angry":
        books_rec.sort_values(by='angry',inplace=True,ascending=False)
    if tone == "Suspenseful":
        books_rec.sort_values(by='fear',inplace=True,ascending=False)
    if tone == "Sad":
        books_rec.sort_values(by='sadness',inplace=True,ascending=False)

    return books_rec


def recommend_books(query,category,tone):

    recommendations = retreive_books(query,category,tone)
    results=[]

    for _,row in recommendations.iterrows():
        
        description = row['description']
        truncated_description_split = description.split()
        truncated_description = ' '.join(truncated_description_split[:30]) + '...'

        authors_split = row['authors'].split(',')
        if len(authors_split) == 2:
            authors = authors_split[0] + ' and ' + authors_split[1]
        elif len(authors_split) > 2:
            authors = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors = row['authors']

        caption = f"{row['title']} by {authors} : {truncated_description}"

        results.append((row['large_thumbnail'], caption))

    return results


categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ['All'] + ['Happy','Surprising','Angry','Suspenseful','Sad']

with gr.Blocks(theme=gr.themes.Glass()) as demo:

    gr.Markdown("""
    # Book Recommendation System
    """)

    with gr.Row():

        user_query = gr.Textbox(label="Enter your query: ",
                                placeholder="e.g. I want to read a book about...")
        
        category_dropdown = gr.Dropdown(choices=categories,
                                        value='All',
                                        label="Select Category")
        
        tone_dropdown = gr.Dropdown(choices=tones,
                                    value='All',
                                    label="Select Tone")

        submit_button = gr.Button("Recommend Books")

    gr.Markdown("""
    ## Recommended Books
    """)

    output = gr.Gallery(label="Recommended Books",
                        columns=8,
                        rows=2)
    
    submit_button.click(fn = recommend_books,
                        inputs=[user_query,category_dropdown,tone_dropdown],
                        outputs=output)
        

if __name__ == "__main__":
    demo.launch()
