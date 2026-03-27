# Smart Text Analyzer - NLP POC

A Streamlit application demonstrating Core NLP Principles:
- Text Preprocessing (lowercasing, punctuation removal, stopwords, lemmatization)
- Tokenization
- TF-IDF Keyword Extraction
- Named Entity Recognition (NER)
- Sentiment Analysis
- Sentence Embeddings + Semantic Similarity Search


## Folder Structure

```
smart_text_analyzer/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md
├── utils/
│   ├── preprocessor.py      # Text cleaning pipeline
│   ├── keywords.py          # TF-IDF keyword extraction
│   ├── ner.py               # Named Entity Recognition
│   ├── sentiment.py         # VADER sentiment analysis
│   └── similarity.py        # Sentence embeddings + cosine similarity
└── data/
    ├── sample_texts.py      # Knowledge base texts for similarity search
    └── learning_notes.py    # Theory notes for the Learning Notes tab
```


## Setup Instructions

### 1. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy's English model
```bash
python -m spacy download en_core_web_sm
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at http://localhost:8501


## Tabs

| Tab | What it shows |
|-----|---------------|
| Text Analyzer | Preprocessing steps, TF-IDF keywords (bar chart), NER entities, Sentiment scores |
| Similarity Search | Semantic search over a knowledge base using sentence embeddings |
| Learning Notes | Full theoretical notes for all 7 Core NLP topics |


## Libraries Used

| Library | Purpose |
|---------|---------|
| spaCy | Tokenization, Lemmatization, NER |
| NLTK | Stopwords, VADER Sentiment |
| scikit-learn | TF-IDF Vectorizer |
| sentence-transformers | Sentence embeddings (all-MiniLM-L6-v2) |
| Streamlit | Web interface |
| Plotly | TF-IDF bar chart visualization |
