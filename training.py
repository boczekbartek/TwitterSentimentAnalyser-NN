# coding: utf-8
from typing import List, Tuple

import numpy as np
from process import *
from nn.MLP import MLPNetwork
from coding import Coding


def get_tweets_data(csv_file: str = "tweets_apple.csv") -> List[Tuple[str, int]]:
    """ Load tweets table from csv. Gets only scored tweets and returns pars with tweet text and score"""
    if not os.path.isfile(csv_file):
        raise FileNotFoundError

    print("Loading from file:", csv_file, sep=" ")
    df = pd.read_csv(csv_file)
    return df.dropna()[['full_text', 'score']]

train_df = get_tweets_data()
train_df['processed_tokens'] = train_df.full_text.apply(tokenize_and_remove_punkt).apply(stem).apply(lem)

coder = Coding()
train_df.processed_tokens.apply(coder.update)

num_features = 30
num_classes = len(train_df.score.unique())
num_hidden_neurons = 100
num_examples = len(train_df)


train_df['coded_tokens'] = (
    train_df
    .processed_tokens
    .apply(lambda l: [coder.encode(tok) for tok in l])
    .apply(partial(pad_or_truncate, target_len=num_features, end=True, pad_value=0))
    .apply(np.array)
)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.coded_tokens.tolist(),
                                                    train_df.score.tolist(), test_size=0.05)

X_train = np.array(X_train)

def simplify(score):
    if score < 3.0:
        return 0
    elif score > 3.0:
        return 2
    else: return 1

y_train = np.array([simplify(i) for i in y_train])