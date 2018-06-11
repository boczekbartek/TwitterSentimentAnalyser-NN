from nltk.stem import PorterStemmer, WordNetLemmatizer

import os
import re
import pandas as pd

from functional import seq
from functools import partial


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

    lemmatizer = lemmatiser_cls()
    return seq(words).map(partial(lemmatizer.lemmatize, pos='v')).list()


def pad_or_truncate(in_list: list, target_len: int, end: bool = True, pad_value :object = 0):
    """
    Fit 'in_list' length to be 'target_len' by adding 0 to the end or beggining depending on 'end' param
    Parameters
    ----------
    in_list : list
        Input list
    target_len : int
        Target length of list
    end : bool
        True - add 'pad_value' to the end of 'in_list' or truncate from end
        False - add 'pad_value' to the beginning of 'in_list' or truncate from beg
    pad_value : object
        Object to add in newly created slots
    Returns
    -------

    """
    if end:
        return in_list[:target_len] + [pad_value] * (target_len - len(in_list))
    else:
        beg_index = len(in_list)-target_len
        beg_index = beg_index if beg_index >= 0 else 0
        return [pad_value] * (target_len - len(in_list)) + in_list[beg_index:]