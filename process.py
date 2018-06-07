from nltk.stem import PorterStemmer, WordNetLemmatizer

import os
import re
import pandas as pd

from functional import seq
from functools import partial


csv_file = "tweets_apple.csv"

os.path.isfile(csv_file)
print("Loading from file:", csv_file, sep=" ")
df = pd.read_csv(csv_file)
df.reindex()


def tokenize_and_remove_punkt(full_text: str) -> list:
    """
    Tokenize and remove punctuation
    Parameters
    ----------
    full_text: str
        text

    Returns
    -------
        list of words
    """
    return re.sub("[^\w]", " ", full_text).split()


def stem(words: list, stemmer_cls=PorterStemmer) -> list:
    """
    Stem every word in list
    Parameters
    ----------
    words : list
        List with words
    stemmer_cls : class
        stemmer class

    Returns
    -------
        List of stemmed words
    """
    stemmer = stemmer_cls()
    return seq(words).map(stemmer.stem).list()


def lem(words: list, lemmatiser_cls=WordNetLemmatizer) -> list:
    """
    Apply lemming on every word in list
    Parameters
    ----------
    words : list
        List with words
    lemmatiser_cls : class
        lemmer class

    Returns
    -------
        List of lemmed words
    """
    lemmatiser = lemmatiser_cls()
    return seq(words).map(partial(lemmatiser.lemmatiser, pos='v')).list()
