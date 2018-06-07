from nltk.stem import PorterStemmer, WordNetLemmatizer

import os
import re
import pandas as pd

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

csv_file = "tweets_apple.csv"

os.path.isfile(csv_file)
print("Loading from file:", csv_file, sep=" ")
df = pd.read_csv(csv_file)
df.reindex()
for index, row in df.iterrows():
    full = row["full_text"]
    wordlist = re.sub("[^\w]", " ",  full).split()
    #print(wordlist)
    for word in wordlist:
        #print(word)
        lem = lemmatiser.lemmatize(word, pos="v")
        stem = stemmer.stem(word)
        d = {'index': [index], 'word': [word], 'stem' :[stem], 'lem':[lem]}
        data = pd.DataFrame(data=d)
        print(data)
        #print("Index: %s, Word: %s, Stem: %s, Lemma: %s" % (index, word, stem, lem))
