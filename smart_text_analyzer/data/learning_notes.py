"""
data/learning_notes.py

All theoretical notes for the 'Learning Notes' tab in the Streamlit app.
Covers every Core NLP Principle topic in structured form.

Each entry has:
    - title: Topic name
    - concept: Plain-English explanation of what it is
    - why: Why it matters / real-world use
    - how: How it works (brief steps)
    - example: A concrete example
    - tools: Libraries used
"""

NOTES = [
    {
        "title": "Text Preprocessing",
        "concept": (
            "Text preprocessing is the first step in every NLP pipeline. "
            "Raw text from users, websites, or documents is noisy — it has uppercase letters, "
            "punctuation, irrelevant words like 'the' and 'is', and different forms of the same word "
            "like 'running', 'ran', 'runs'. Preprocessing cleans all of this so the model "
            "works with clean, standardized input."
        ),
        "why": (
            "A computer treats 'Apple', 'apple', and 'APPLE' as three completely different words "
            "without preprocessing. Cleaning ensures consistency and removes noise that would "
            "confuse or bloat the model."
        ),
        "how": [
            "Lowercasing — convert all text to lowercase ('Dog' becomes 'dog')",
            "Punctuation removal — strip out !, ., ?, ##, etc.",
            "Stopword removal — remove common words with little meaning: 'the', 'is', 'in', 'a', 'an'",
            "Stemming — chop word endings to get rough root: 'running' -> 'run' (fast but crude)",
            "Lemmatization — use a dictionary to find the actual root: 'better' -> 'good' (accurate, preferred)",
        ],
        "example": (
            "Input:  'The Scientists were Studying the Effects of Machine Learning!!'\n"
            "Output: ['scientist', 'study', 'effect', 'machine', 'learn']"
        ),
        "tools": "NLTK (stopwords, stemming), spaCy (lemmatization)",
        "learning_outcome": "Used before every NLP task — foundational for RAG, PII detection, and monitoring.",
    },
    {
        "title": "Tokenization",
        "concept": (
            "Tokenization is the process of splitting text into smaller units called tokens. "
            "A token is usually a word, but can also be a sentence, or even a subword piece. "
            "It is the very first operation any NLP model performs on text — you cannot analyze "
            "text as a whole blob, you need to break it into pieces first."
        ),
        "why": (
            "Computers cannot natively understand a sentence as a sequence of meaning. "
            "Tokenization gives the model individual units to process one by one. "
            "Modern LLMs (GPT, BERT) also use tokenization internally — they process "
            "tokens, not characters or full sentences."
        ),
        "how": [
            "Word Tokenization — split by whitespace and punctuation: 'I love NLP' -> ['I', 'love', 'NLP']",
            "Sentence Tokenization — split a paragraph into sentences",
            "Subword Tokenization — used in BERT/GPT: 'unhappiness' -> ['un', '##happi', '##ness'] (handles unknown words)",
        ],
        "example": (
            "Input:  'Natural language processing is important.'\n"
            "Word tokens: ['Natural', 'language', 'processing', 'is', 'important', '.']\n"
            "spaCy also gives POS tags: Natural=ADJ, language=NOUN, processing=NOUN, etc."
        ),
        "tools": "NLTK (word_tokenize, sent_tokenize), spaCy (doc.sents, doc tokens)",
        "learning_outcome": "Tokenization is how LLMs read your prompts — directly relevant to Prompt Engineering.",
    },
    {
        "title": "TF-IDF (Term Frequency – Inverse Document Frequency)",
        "concept": (
            "TF-IDF is a numerical method to measure how important a word is in a specific document "
            "relative to a collection of documents (called a corpus). It rewards words that appear "
            "frequently in one document but are rare across the whole collection — those are the meaningful keywords."
        ),
        "why": (
            "Simple word counting gives high scores to words like 'the', 'is', 'and' — "
            "these appear everywhere and tell us nothing. TF-IDF automatically penalizes such "
            "common words and surfaces the words that actually characterize a document."
        ),
        "how": [
            "TF (Term Frequency) = count of word in this document / total words in document",
            "IDF (Inverse Document Frequency) = log(total documents / documents containing the word)",
            "TF-IDF score = TF x IDF",
            "High TF-IDF = word is frequent here but rare elsewhere = important keyword",
            "Low TF-IDF  = word is common everywhere (like 'the') = not important",
        ],
        "example": (
            "3 articles: one about cricket, two about football.\n"
            "'the'     -> appears in all 3 -> IDF is low -> low TF-IDF (not important)\n"
            "'wicket'  -> appears only in cricket article -> IDF is high -> high TF-IDF (important keyword!)"
        ),
        "tools": "scikit-learn TfidfVectorizer",
        "learning_outcome": "Keyword extraction is used in summarization, search indexing, and document analysis pipelines.",
    },
    {
        "title": "Embeddings",
        "concept": (
            "Embeddings represent text — a word, sentence, or paragraph — as a list of numbers (a vector). "
            "What makes embeddings special is that texts with similar meanings get similar vectors. "
            "This allows computers to understand semantic similarity, not just keyword overlap."
        ),
        "why": (
            "TF-IDF and simple counting treat every word as completely separate. They cannot know that "
            "'car' and 'automobile' are the same, or that 'happy' and 'joyful' are similar. "
            "Embeddings encode actual meaning, making them essential for modern NLP tasks."
        ),
        "how": [
            "A neural network is trained on millions of text examples",
            "It learns to place similar words/sentences close together in vector space",
            "The result: each text is mapped to a dense vector (e.g., 384 numbers)",
            "Cosine similarity between two vectors measures how similar they are (0 = different, 1 = same)",
            "This is used for: semantic search, RAG retrieval, document clustering, recommendations",
        ],
        "example": (
            "vector('king') - vector('man') + vector('woman') ≈ vector('queen')  [Word2Vec famous example]\n\n"
            "'I love coding'       -> [0.23, -0.45, 0.88, ...]  <- similar vectors\n"
            "'Python is my hobby' -> [0.21, -0.43, 0.85, ...]  <-\n"
            "'The sky is blue'    -> [-0.55, 0.12, -0.30, ...]  <- very different"
        ),
        "tools": "sentence-transformers (all-MiniLM-L6-v2), spaCy vectors, OpenAI embeddings API",
        "learning_outcome": "Embeddings are the core of RAG systems (retrieval step). Also used in semantic monitoring and drift detection.",
    },
    {
        "title": "Named Entity Recognition (NER)",
        "concept": (
            "Named Entity Recognition is an NLP task that identifies and classifies named real-world objects "
            "in text. These include people's names, company names, locations, dates, monetary values, and more. "
            "NER turns unstructured text into structured information you can extract and use."
        ),
        "why": (
            "Imagine processing thousands of news articles and needing to know which companies are mentioned, "
            "which people are involved, and what dates are referenced. NER automates this completely. "
            "It is also critical for PII (Personally Identifiable Information) detection."
        ),
        "how": [
            "spaCy's NER model reads each word in context (not just individually)",
            "It uses a trained neural network that has seen millions of labeled examples",
            "It outputs entity spans with labels: PERSON, ORG, GPE, DATE, MONEY, etc.",
            "GPE = Geopolitical Entity (country, city, state)",
            "NORP = Nationalities, religious, or political groups",
        ],
        "example": (
            "Input: 'Elon Musk founded SpaceX in 2002 in California with $100 million.'\n"
            "Output:\n"
            "  Elon Musk  -> PERSON\n"
            "  SpaceX     -> ORG\n"
            "  2002       -> DATE\n"
            "  California -> GPE\n"
            "  $100 million -> MONEY"
        ),
        "tools": "spaCy (en_core_web_sm, en_core_web_lg), Hugging Face transformers (for better accuracy)",
        "learning_outcome": "NER is directly used in PII detection (find names, locations, IDs) and guardrails (redact sensitive entities before sending to LLMs).",
    },
    {
        "title": "Sentiment Analysis",
        "concept": (
            "Sentiment Analysis determines the emotional tone of a piece of text. "
            "It classifies text as Positive, Negative, or Neutral. "
            "More advanced models also detect emotions (joy, anger, fear, surprise) "
            "or specific aspects of sentiment (product quality vs. customer service)."
        ),
        "why": (
            "Used everywhere — product review analysis, social media monitoring, customer feedback, "
            "brand reputation tracking, and even financial news analysis (positive news = stock up?)."
        ),
        "how": [
            "Lexicon-based (VADER): uses a dictionary of words scored with sentiment values",
            "  'love' = +3.0, 'hate' = -3.0, 'okay' = +0.5",
            "  VADER also handles punctuation (!!), capitalization (GREAT), and emoji",
            "ML-based (BERT, etc.): a neural network trained on labeled sentiment data",
            "  More accurate but heavier; better for complex sentences with sarcasm",
            "VADER compound score: ranges from -1.0 (very negative) to +1.0 (very positive)",
            "  >= 0.05 = Positive, <= -0.05 = Negative, otherwise Neutral",
        ],
        "example": (
            "Input: 'I absolutely LOVE this new feature!! It works perfectly.'\n"
            "  compound: +0.83  -> Positive\n\n"
            "Input: 'This is the worst experience I have ever had.'\n"
            "  compound: -0.68  -> Negative\n\n"
            "Input: 'The meeting is scheduled for 3pm.'\n"
            "  compound: +0.00  -> Neutral"
        ),
        "tools": "NLTK VADER, TextBlob, Hugging Face transformers (distilbert-sentiment)",
        "learning_outcome": "Sentiment analysis is used in monitoring model outputs — detecting if responses drift negative or become inappropriate.",
    },
    {
        "title": "Text Similarity and Semantic Search",
        "concept": (
            "Text similarity measures how semantically close two pieces of text are. "
            "Unlike keyword matching (does word X appear in document Y?), semantic similarity "
            "understands meaning. Two sentences can be similar even if they share zero words in common. "
            "Semantic search uses this to find the most relevant documents for a query."
        ),
        "why": (
            "This is the foundation of RAG (Retrieval Augmented Generation). "
            "When a user asks a question, RAG retrieves the most semantically similar documents "
            "from a knowledge base and passes them to the LLM as context. "
            "Without semantic similarity, retrieval would only work on exact keyword matches."
        ),
        "how": [
            "Encode query and all candidate documents into embeddings (vectors)",
            "Compute cosine similarity between the query vector and each document vector",
            "Rank documents by similarity score (highest = most relevant)",
            "Return top-K results — these become the context for the LLM in RAG",
            "Cosine similarity = dot product of vectors / (magnitude of v1 * magnitude of v2)",
        ],
        "example": (
            "Query: 'What is deep learning?'\n\n"
            "Candidates:\n"
            "  [0.87] 'Deep learning uses neural networks with many layers.'  <- Best match\n"
            "  [0.72] 'Machine learning is a type of artificial intelligence.' <- Related\n"
            "  [0.12] 'India won the cricket world cup.'  <- Unrelated\n\n"
            "The top result would be passed to an LLM as context in a RAG pipeline."
        ),
        "tools": "sentence-transformers, FAISS (for large-scale vector search), Pinecone, ChromaDB",
        "learning_outcome": "This IS the retrieval step in RAG.",
    },
]
