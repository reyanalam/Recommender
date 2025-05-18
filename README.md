# Book Recommender System

A sophisticated book recommendation system that uses natural language processing and emotion analysis to suggest books based on user queries, categories, and desired emotional tones.

## Features

- **Semantic Search**: Find books based on natural language queries
- **Category Filtering**: Filter recommendations by book categories
- **Emotional Tone Selection**: Get recommendations based on emotional content (Happy, Surprising, Angry, Suspenseful, Sad)
- **Interactive UI**: User-friendly Gradio interface for easy interaction
- **Book Details**: View book covers, titles, authors, and descriptions

## Technical Stack

- **Python**: Core programming language
- **LangChain**: For document processing and embeddings
- **HuggingFace**: Using sentence-transformers for text embeddings
- **Chroma**: Vector database for similarity search
- **Gradio**: Web interface
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Book-Recommender
```

2. Create and activate a virtual environment:
```bash
conda create -m venv python==3.11
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python frontent.py
```

2. Open your web browser and navigate to the local URL provided by Gradio (typically http://127.0.0.1:7860)

3. Use the interface to:
   - Enter your book query
   - Select a category
   - Choose an emotional tone
   - Click "Recommend Books" to get suggestions

## Data Structure

The system uses two main data sources:
- `books_with_emotion.csv`: Contains book metadata and emotional analysis
- `taggged_description.txt`: Contains processed book descriptions for semantic search

## How It Works

1. **Query Processing**: User input is processed using semantic search
2. **Initial Filtering**: System retrieves initial set of relevant books
3. **Category Filtering**: Results are filtered by selected category
4. **Emotional Sorting**: Books are sorted based on emotional content
5. **Result Display**: Final recommendations are displayed with book covers and details

## Contributing

Feel free to submit issues and enhancement requests!



