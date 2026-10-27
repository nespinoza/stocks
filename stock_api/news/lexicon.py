"""Loughran–McDonald dictionary baseline for headline tone.

The LM Master Dictionary is free for academic research; commercial use needs a
license from its authors (https://sraf.nd.edu). It is downloaded at run time
into the git-ignored data_store/, never committed.
"""
import re
from pathlib import Path

import pandas as pd
import requests

LM_URL = 'https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t'
NEGATIONS = {'no', 'not', 'none', 'neither', 'never', 'nobody', "isn't", "wasn't", "aren't", "doesn't", "didn't", "won't"}


def load_lm(path='data_store/lexicon/LM_MasterDictionary.csv'):
    """Returns (positive words, negative words, uncertainty words) as upper-case sets."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(LM_URL, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b'Word,'):
            raise RuntimeError('Unexpected LM dictionary download; fetch it manually from sraf.nd.edu')
        path.write_bytes(response.content)
    table = pd.read_csv(path, usecols=['Word', 'Positive', 'Negative', 'Uncertainty'])
    pick = lambda column: set(table.loc[table[column] > 0, 'Word'].astype(str).str.upper())
    return pick('Positive'), pick('Negative'), pick('Uncertainty')


def tone(text, positive, negative, uncertainty=frozenset()):
    """(pos − neg)/(pos + neg) with simple negation flipping of positive words (LM 2011)."""
    words = re.findall(r"[A-Za-z']+", text)
    pos = neg = unc = 0
    for i, word in enumerate(words):
        upper = word.upper()
        negated = any(w.lower() in NEGATIONS for w in words[max(0, i-3):i])
        if upper in positive:
            pos, neg = (pos, neg+1) if negated else (pos+1, neg)
        elif upper in negative:
            neg += 1
        if upper in uncertainty:
            unc += 1
    return dict(lm_tone=(pos-neg)/(pos+neg) if pos+neg else 0., lm_positive=pos, lm_negative=neg, lm_uncertainty=unc,
                lm_words=len(words))
