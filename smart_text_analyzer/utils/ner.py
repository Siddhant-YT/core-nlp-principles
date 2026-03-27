"""
utils/ner.py

Named Entity Recognition (NER) using spaCy.

What is NER?
    NER identifies and classifies real-world objects in text.
    For example, in "Elon Musk founded Tesla in 2002 in California":
        - 'Elon Musk' → PERSON
        - 'Tesla'     → ORG
        - '2002'      → DATE
        - 'California'→ GPE (Geopolitical Entity)

How spaCy's NER works:
    spaCy uses a trained neural network model that has learned
    to recognize entity patterns from millions of real texts.
    It looks at each word and its surrounding context to decide the label.
"""

import spacy
from collections import defaultdict

# Load spaCy model (shared with preprocessor — no double loading in practice,
# but we import it here to keep utils self-contained)
nlp = spacy.load("en_core_web_sm")

# Human-readable labels for each entity type spaCy can detect
ENTITY_LABELS = {
    "PERSON":   "Person",
    "ORG":      "Organization",
    "GPE":      "Country / City / State",
    "LOC":      "Location",
    "DATE":     "Date",
    "TIME":     "Time",
    "MONEY":    "Money / Currency",
    "PERCENT":  "Percentage",
    "PRODUCT":  "Product",
    "EVENT":    "Event",
    "LAW":      "Law / Policy",
    "NORP":     "Nationality / Group",
    "WORK_OF_ART": "Work of Art",
    "CARDINAL": "Number",
    "QUANTITY": "Quantity",
    "ORDINAL":  "Ordinal (first, second...)",
    "LANGUAGE": "Language",
    "FAC":      "Facility (building, airport...)",
}


def extract_entities(text: str) -> dict:
    """
    Extract all named entities from text and group them by entity type.

    Args:
        text: Raw or lightly cleaned input text
              (we use original text here — NER works better on original casing)

    Returns:
        dict mapping entity label -> list of unique entity strings
        Example:
        {
            "PERSON": ["Elon Musk", "Tim Cook"],
            "ORG":    ["Tesla", "Apple"],
            "DATE":   ["2002", "last Monday"]
        }
    """
    # Process the text through spaCy's pipeline
    # (includes tokenizer, tagger, parser, and NER in one pass)
    doc = nlp(text)

    # Group entities by their label
    entity_groups = defaultdict(list)
    for ent in doc.ents:
        # ent.text  = the actual text span (e.g., "Elon Musk")
        # ent.label_ = the entity category (e.g., "PERSON")
        entity_groups[ent.label_].append(ent.text)

    # Remove duplicate entities within each group while preserving order
    deduplicated = {
        label: list(dict.fromkeys(entities))   # dict.fromkeys preserves order and removes dupes
        for label, entities in entity_groups.items()
    }

    return deduplicated


def get_entity_flat_list(text: str) -> list[dict]:
    """
    Return a flat list of all entities with their position in the text.
    Useful for highlighting entities in a display.

    Returns:
        List of dicts: [{"text": "Elon Musk", "label": "PERSON", "start": 0, "end": 9}, ...]
    """
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "readable_label": ENTITY_LABELS.get(ent.label_, ent.label_),
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
