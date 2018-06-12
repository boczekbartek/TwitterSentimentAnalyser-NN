# coding: utf-8
import os
from typing import List, Tuple

import pandas as pd


def get_train_data(csv_file: str = "tweets_apple.csv") -> List[Tuple[str, int]]:
    """ Load tweets table from csv. Gets only scored tweets and returns pars with tweet text and score"""
    if not os.path.isfile(csv_file):
        raise FileNotFoundError

    print("Loading from file:", csv_file, sep=" ")
    df = pd.read_csv(csv_file)
    return df.dropna()[['full_text', 'score']]


def simplify(score):
    """ Reduce number of classes """
    if score < 3.0:
        return 0
    elif score > 3.0:
        return 2
    else: return 1
