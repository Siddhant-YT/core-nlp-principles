"""
app.py — Smart Text Analyzer
Entry point for the Streamlit application.

Tabs:
    1. Analyzer       — Run all NLP analysis on user input text
    2. Similarity     — Compare input text against a knowledge base using embeddings
    3. Learning Notes — Theory notes for all Core NLP Principles topics

Run with:
    streamlit run app.py
"""

import sys
import os

# Add project root to Python path so we can import from utils/ and data/
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import our utility modules
from utils.preprocessor import full_preprocess
from utils.keywords import extract_keywords
from utils.ner import extract_entities, ENTITY_LABELS
from utils.sentiment import analyze_sentiment
from utils.similarity import load_embedding_model, rank_by_similarity

# Import data
from data.sample_texts import SAMPLE_TEXTS
from data.learning_notes import NOTES


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Text Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a clean, professional look
# We use a dark-accent theme with monospace touches to fit the NLP/tech context
st.markdown("""
<style>
    /* Import a clean, technical-looking font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* App background */
    .stApp {
        background-color: #0f1117;
        color: #e2e8f0;
    }

    /* Main block padding */
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1a1d27;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: #8892a4;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #2d3147 !important;
        color: #e2e8f0 !important;
    }

    /* Text area */
    .stTextArea textarea {
        background: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 8px;
        color: #e2e8f0;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
    }

    /* Buttons */
    .stButton > button {
        background: #3b4fd8;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        padding: 0.5rem 2rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #4e62e8;
        color: white;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 8px;
        padding: 12px;
    }

    /* Section card */
    .section-card {
        background: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    /* Entity badge */
    .entity-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 3px 3px;
        font-family: 'DM Sans', sans-serif;
    }

    /* Keyword pill */
    .keyword-pill {
        display: inline-block;
        background: #1e2235;
        border: 1px solid #3b4fd8;
        color: #a5b4fc;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 3px;
        font-size: 0.82rem;
        font-family: 'DM Mono', monospace;
    }

    /* Note card */
    .note-card {
        background: #1a1d27;
        border-left: 3px solid #3b4fd8;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin-bottom: 8px;
    }

    /* Subheader override */
    h3 {
        color: #c7d2fe;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Divider */
    hr {
        border-color: #2d3147;
    }

    /* Similarity score bar bg */
    .sim-bar-bg {
        background: #1e2235;
        border-radius: 4px;
        height: 8px;
        margin-top: 4px;
    }

    /* Step box for preprocessing */
    .step-box {
        background: #12151e;
        border: 1px solid #252840;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .step-label {
        color: #6366f1;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (cached — loads once for the entire session)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading NLP models..."):
    embedding_model = load_embedding_model()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## Smart Text Analyzer")
st.markdown(
    "<p style='color:#64748b;font-size:0.9rem;margin-top:-8px'>"
    "Core NLP Principles POC &nbsp;|&nbsp; "
    "Preprocessing &middot; TF-IDF &middot; NER &middot; Sentiment &middot; Embeddings"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_analyzer, tab_similarity, tab_notes = st.tabs([
    "Text Analyzer",
    "Similarity Search",
    "Learning Notes",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: TEXT ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

with tab_analyzer:

    # Default text to show on load so users see results immediately
    DEFAULT_TEXT = (
        "Apple CEO Tim Cook visited New Delhi in April 2023 and met Prime Minister Narendra Modi. "
        "Apple announced plans to invest $1 billion in manufacturing operations across India. "
        "The Cupertino-based technology company opened its first retail stores in Mumbai and Delhi. "
        "Analysts at Goldman Sachs predict Apple's India revenue could reach $50 billion by 2030, "
        "making it one of the most important growth markets for the company."
    )

    st.markdown("#### Input Text")
    user_text = st.text_area(
        label="Enter any text to analyze",
        value=DEFAULT_TEXT,
        height=150,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("Analyze", type="primary")

    # Run analysis when button clicked OR on first load with default text
    if analyze_clicked or True:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # ── Run all NLP modules ──────────────────────────────────────────
            preprocess_result = full_preprocess(user_text)
            keywords          = extract_keywords(preprocess_result["cleaned_string"], top_n=10)
            entities          = extract_entities(user_text)    # use original text for NER
            sentiment         = analyze_sentiment(user_text)

            st.markdown("---")

            # ── ROW 1: Preprocessing steps ───────────────────────────────────
            st.markdown("### Preprocessing Pipeline")
            st.markdown(
                "<p style='color:#64748b;font-size:0.82rem;margin-top:-10px'>"
                "Raw text goes through these steps before any analysis.</p>",
                unsafe_allow_html=True
            )

            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown(
                    f"<div class='step-box'>"
                    f"<div class='step-label'>Step 1 — Lowercase</div>"
                    f"{preprocess_result['lowercased'][:200]}{'...' if len(preprocess_result['lowercased']) > 200 else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='step-box'>"
                    f"<div class='step-label'>Step 2 — Punctuation Removed</div>"
                    f"{preprocess_result['no_punctuation'][:200]}{'...' if len(preprocess_result['no_punctuation']) > 200 else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_b:
                token_display = "  |  ".join(preprocess_result["tokens"][:20])
                if len(preprocess_result["tokens"]) > 20:
                    token_display += "  |  ..."
                st.markdown(
                    f"<div class='step-box'>"
                    f"<div class='step-label'>Step 3 — Lemmatized Tokens (stopwords removed)</div>"
                    f"{token_display}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='step-box'>"
                    f"<div class='step-label'>Token Count</div>"
                    f"Original words: {len(user_text.split())} &nbsp;&nbsp; "
                    f"After cleaning: {len(preprocess_result['tokens'])}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # ── ROW 2: Keywords + Sentiment side by side ─────────────────────
            col_kw, col_sent = st.columns([3, 2])

            with col_kw:
                st.markdown("### TF-IDF Keywords")
                st.markdown(
                    "<p style='color:#64748b;font-size:0.82rem;margin-top:-10px'>"
                    "Words scored by importance in this document (high score = meaningful keyword).</p>",
                    unsafe_allow_html=True
                )

                if keywords:
                    # Build a horizontal bar chart with Plotly
                    words  = [kw[0] for kw in keywords][::-1]   # reverse so top word is at top
                    scores = [kw[1] for kw in keywords][::-1]

                    fig = go.Figure(go.Bar(
                        x=scores,
                        y=words,
                        orientation="h",
                        marker=dict(
                            color=scores,
                            colorscale=[[0, "#1e2235"], [1, "#4e62e8"]],
                            showscale=False,
                        ),
                        text=[f"{s:.4f}" for s in scores],
                        textposition="outside",
                        textfont=dict(color="#94a3b8", size=11),
                    ))
                    fig.update_layout(
                        margin=dict(l=0, r=60, t=10, b=10),
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        yaxis=dict(
                            showgrid=False,
                            tickfont=dict(color="#c7d2fe", family="DM Mono", size=12),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough content to extract keywords.")

            with col_sent:
                st.markdown("### Sentiment")
                st.markdown(
                    "<p style='color:#64748b;font-size:0.82rem;margin-top:-10px'>"
                    "VADER sentiment scores — lexicon-based approach.</p>",
                    unsafe_allow_html=True
                )

                # Color coding per label
                label_colors = {
                    "Positive": ("#16a34a", "#dcfce7"),
                    "Negative": ("#dc2626", "#fee2e2"),
                    "Neutral":  ("#ca8a04", "#fef9c3"),
                }
                fg, bg = label_colors.get(sentiment["label"], ("#6366f1", "#e0e7ff"))

                st.markdown(
                    f"<div style='background:#1a1d27;border:1px solid #2d3147;border-radius:10px;"
                    f"padding:20px 24px;margin-top:8px'>"
                    f"<div style='font-size:2rem;font-weight:700;color:{fg}'>{sentiment['label']}</div>"
                    f"<div style='color:#64748b;font-size:0.8rem;margin-top:4px'>Overall sentiment</div>"
                    f"<hr style='border-color:#2d3147;margin:14px 0'>"
                    f"<table style='width:100%;color:#94a3b8;font-size:0.85rem'>"
                    f"<tr><td>Compound score</td><td style='text-align:right;color:#e2e8f0;font-family:DM Mono'>{sentiment['compound']}</td></tr>"
                    f"<tr><td style='color:#16a34a'>Positive fraction</td><td style='text-align:right;font-family:DM Mono'>{sentiment['positive']}</td></tr>"
                    f"<tr><td style='color:#dc2626'>Negative fraction</td><td style='text-align:right;font-family:DM Mono'>{sentiment['negative']}</td></tr>"
                    f"<tr><td>Neutral fraction</td><td style='text-align:right;font-family:DM Mono'>{sentiment['neutral']}</td></tr>"
                    f"</table>"
                    f"<div style='color:#475569;font-size:0.72rem;margin-top:14px'>"
                    f"Compound: +1.0 = very positive, -1.0 = very negative</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # ── ROW 3: Named Entities ────────────────────────────────────────
            st.markdown("### Named Entity Recognition")
            st.markdown(
                "<p style='color:#64748b;font-size:0.82rem;margin-top:-10px'>"
                "Real-world objects identified and classified in the text.</p>",
                unsafe_allow_html=True
            )

            # Color palette for different entity types
            ENTITY_COLORS = {
                "PERSON":  ("#3b82f6", "#1e3a5f"),
                "ORG":     ("#8b5cf6", "#2e1f5e"),
                "GPE":     ("#06b6d4", "#0c3d4a"),
                "LOC":     ("#14b8a6", "#0a3535"),
                "DATE":    ("#f59e0b", "#3d2c06"),
                "TIME":    ("#f97316", "#3d1f06"),
                "MONEY":   ("#10b981", "#0a3322"),
                "PERCENT": ("#84cc16", "#243506"),
                "PRODUCT": ("#ec4899", "#3d0f28"),
                "EVENT":   ("#f43f5e", "#3d0a14"),
                "NORP":    ("#6366f1", "#1e1f4a"),
                "CARDINAL":("#94a3b8", "#1e2435"),
            }

            if entities:
                # Layout: up to 3 entity groups per row
                entity_items = list(entities.items())
                cols = st.columns(min(len(entity_items), 3))

                for i, (label, ent_list) in enumerate(entity_items):
                    col_idx = i % 3
                    fg_color, bg_color = ENTITY_COLORS.get(label, ("#94a3b8", "#1e2435"))
                    readable = ENTITY_LABELS.get(label, label)

                    with cols[col_idx]:
                        badges_html = "".join(
                            f"<span class='entity-badge' style='background:{bg_color};color:{fg_color};border:1px solid {fg_color}40'>{e}</span>"
                            for e in ent_list
                        )
                        st.markdown(
                            f"<div style='background:#1a1d27;border:1px solid #2d3147;border-radius:8px;"
                            f"padding:12px 16px;margin-bottom:10px'>"
                            f"<div style='color:{fg_color};font-size:0.72rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>"
                            f"{readable}</div>"
                            f"{badges_html}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.info("No named entities detected in the text. Try a text that mentions people, organizations, or places.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────────────────────────

with tab_similarity:

    st.markdown("#### Semantic Similarity Search")
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem'>"
        "Enter a query and find the most semantically similar texts from the knowledge base. "
        "This is the core mechanism behind RAG (Retrieval Augmented Generation) — "
        "embeddings convert text to vectors, then cosine similarity ranks relevance.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Query Text**")
        query_text = st.text_area(
            "query_input",
            value="What is deep learning and how does it relate to AI?",
            height=100,
            label_visibility="collapsed",
        )

        st.markdown("**Knowledge Base**")
        st.markdown(
            "<p style='color:#64748b;font-size:0.78rem;margin-top:-6px'>"
            "One document per line. Edit freely.</p>",
            unsafe_allow_html=True
        )
        kb_text = st.text_area(
            "kb_input",
            value="\n".join(SAMPLE_TEXTS),
            height=280,
            label_visibility="collapsed",
        )

        search_clicked = st.button("Find Similar Documents", type="primary")

    with col_right:
        st.markdown("**Results — Ranked by Similarity**")

        if search_clicked:
            if not query_text.strip():
                st.warning("Please enter a query.")
            else:
                # Split the knowledge base into individual documents
                candidates = [line.strip() for line in kb_text.strip().split("\n") if line.strip()]

                if not candidates:
                    st.warning("Please add some documents to the knowledge base.")
                else:
                    with st.spinner("Computing embeddings..."):
                        results = rank_by_similarity(query_text.strip(), candidates, embedding_model)

                    # Display each result with a visual score bar
                    for rank, result in enumerate(results, 1):
                        score = result["score"]
                        text  = result["text"]

                        # Determine color category based on score
                        if score >= 0.6:
                            bar_color  = "#4e62e8"
                            score_color = "#a5b4fc"
                            rank_label = "High match"
                        elif score >= 0.35:
                            bar_color  = "#f59e0b"
                            score_color = "#fcd34d"
                            rank_label = "Moderate"
                        else:
                            bar_color  = "#475569"
                            score_color = "#64748b"
                            rank_label = "Low match"

                        # Score bar width as percentage (max bar at score=1.0)
                        bar_pct = int(score * 100)

                        st.markdown(
                            f"<div style='background:#1a1d27;border:1px solid #2d3147;border-radius:8px;"
                            f"padding:12px 16px;margin-bottom:8px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
                            f"<span style='color:#94a3b8;font-size:0.78rem'>#{rank} &nbsp; {rank_label}</span>"
                            f"<span style='color:{score_color};font-family:DM Mono;font-size:0.85rem;font-weight:600'>{score:.4f}</span>"
                            f"</div>"
                            f"<div style='background:#12151e;border-radius:3px;height:5px;margin-bottom:10px'>"
                            f"<div style='background:{bar_color};height:5px;border-radius:3px;width:{bar_pct}%'></div>"
                            f"</div>"
                            f"<div style='color:#e2e8f0;font-size:0.83rem;line-height:1.5'>{text}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
        else:
            # Placeholder state
            st.markdown(
                "<div style='background:#1a1d27;border:1px dashed #2d3147;border-radius:8px;"
                "padding:40px;text-align:center;color:#475569'>"
                "<div style='font-size:0.9rem'>Enter a query and click 'Find Similar Documents'</div>"
                "<div style='font-size:0.78rem;margin-top:8px'>"
                "Results will be ranked by cosine similarity of sentence embeddings</div>"
                "</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Explanation section
    st.markdown("**How does this work?**")
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.markdown(
            "<div class='note-card'>"
            "<div style='color:#c7d2fe;font-size:0.85rem;font-weight:600;margin-bottom:6px'>Step 1: Encode</div>"
            "<div style='color:#94a3b8;font-size:0.82rem'>"
            "The query and every document in the knowledge base are each converted into a 384-dimensional vector "
            "using a pre-trained sentence transformer model.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with col_e2:
        st.markdown(
            "<div class='note-card'>"
            "<div style='color:#c7d2fe;font-size:0.85rem;font-weight:600;margin-bottom:6px'>Step 2: Compare</div>"
            "<div style='color:#94a3b8;font-size:0.82rem'>"
            "Cosine similarity is computed between the query vector and each document vector. "
            "Texts with similar meaning point in the same direction in vector space, giving a high score.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with col_e3:
        st.markdown(
            "<div class='note-card'>"
            "<div style='color:#c7d2fe;font-size:0.85rem;font-weight:600;margin-bottom:6px'>Step 3: Rank</div>"
            "<div style='color:#94a3b8;font-size:0.82rem'>"
            "Documents are sorted by score (0 = unrelated, 1 = identical meaning). "
            "In a RAG system, the top results are passed to the LLM as context for answering.</div>"
            "</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: LEARNING NOTES
# ─────────────────────────────────────────────────────────────────────────────

with tab_notes:

    st.markdown("#### Core NLP Principles — Study Notes")
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem'>"
        "Complete theoretical reference for all topics covered in this POC. "
        "Each note covers concept, intuition, how it works, examples and tools.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Index of topics for quick navigation
    topic_names = [note["title"] for note in NOTES]
    st.markdown(
        "<div style='background:#1a1d27;border:1px solid #2d3147;border-radius:8px;padding:14px 20px;margin-bottom:20px'>"
        "<div style='color:#6366f1;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>Topics Covered</div>"
        + "".join(
            f"<span style='background:#1e2235;border:1px solid #3b4fd8;color:#a5b4fc;"
            f"border-radius:4px;padding:3px 12px;margin:3px;display:inline-block;font-size:0.8rem'>"
            f"{i+1}. {name}</span>"
            for i, name in enumerate(topic_names)
        )
        + "</div>",
        unsafe_allow_html=True
    )

    # Render each note as an expandable section
    for note in NOTES:
        with st.expander(f"  {note['title']}", expanded=False):

            # Concept
            st.markdown(
                f"<div class='note-card'>"
                f"<div style='color:#6366f1;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:0.06em;margin-bottom:6px'>What is it?</div>"
                f"<div style='color:#cbd5e1;font-size:0.88rem;line-height:1.7'>{note['concept']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            col_why, col_tools = st.columns([2, 1])

            with col_why:
                # Why it matters
                st.markdown(
                    f"<div class='note-card'>"
                    f"<div style='color:#6366f1;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:0.06em;margin-bottom:6px'>Why does it matter?</div>"
                    f"<div style='color:#94a3b8;font-size:0.85rem;line-height:1.6'>{note['why']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_tools:
                # Tools
                st.markdown(
                    f"<div class='note-card'>"
                    f"<div style='color:#6366f1;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:0.06em;margin-bottom:6px'>Tools</div>"
                    f"<div style='color:#94a3b8;font-size:0.83rem;line-height:1.6;font-family:DM Mono'>{note['tools']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # How it works — step by step
            steps_html = "".join(
                f"<div style='display:flex;gap:10px;margin-bottom:6px;align-items:flex-start'>"
                f"<span style='color:#3b4fd8;font-weight:700;font-size:0.8rem;min-width:18px;margin-top:1px'>{i+1}.</span>"
                f"<span style='color:#94a3b8;font-size:0.85rem;line-height:1.5'>{step}</span>"
                f"</div>"
                for i, step in enumerate(note["how"])
            )
            st.markdown(
                f"<div class='note-card'>"
                f"<div style='color:#6366f1;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:0.06em;margin-bottom:10px'>How it works</div>"
                f"{steps_html}"
                f"</div>",
                unsafe_allow_html=True
            )

            # Example
            st.markdown(
                f"<div class='note-card'>"
                f"<div style='color:#6366f1;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:0.06em;margin-bottom:6px'>Example</div>"
                f"<pre style='color:#a5b4fc;font-family:DM Mono;font-size:0.8rem;"
                f"background:#12151e;border-radius:6px;padding:12px;white-space:pre-wrap;margin:0'>"
                f"{note['example']}</pre>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Learning Outcome
            st.markdown(
                f"<div style='background:#0f1d2e;border:1px solid #1e3a5f;border-radius:8px;"
                f"padding:12px 16px;margin-top:4px'>"
                f"<span style='color:#3b82f6;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:0.06em'>Learning Outcome &nbsp;</span>"
                f"<span style='color:#93c5fd;font-size:0.85rem'>{note['learning_outcome']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#334155;font-size:0.78rem;padding:8px'>"
        "Smart Text Analyzer &nbsp;|&nbsp; Core NLP Principles POC"
        "</div>",
        unsafe_allow_html=True
    )
