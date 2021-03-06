from collections import Counter


class Coding:
    def __init__(self):
        self.word_dict = dict()
        self.rev_dict = dict()
        self.occurrences = Counter()
        self.max_code = 0
        self.final_dict = None

    def update(self, words: list) -> dict:
        """
        Update dicts from list of words, set to self.word_dict. There are
        Parameters
        ----------
        words: list
            list of words to create dict
        Returns
        -------
            dict of words
        """
        for word in words:
            self.update_dict(word)

        return self.word_dict

    def update_dict(self, word) -> int:
        """
        Add value to dicts
        Parameters
        ----------
        word: str
            word to add to dict
        Returns
        -------
            index of added word, or index of this word if existed in dict previously
        """
        self.add_occurrence(word)
        if word not in self.word_dict:
            new_code = self.max_code + 1
            self.word_dict[word] = new_code
            self.rev_dict[new_code] = word
            self.max_code = new_code

        return self.word_dict[word]

    def add_occurrence(self, word: str) -> int:
        """
        Add occurrence of word to occurrences dict
        Parameters
        ----------
        word: str
            word
        Returns
        -------
            Number of occurrences for this word
        """
        if word not in self.word_dict:
            self.occurrences[word] = 1
        else:
            self.occurrences[word] += 1

        return self.occurrences[word]

    def decode(self, value: int) -> str:
        """
        Decode int to word using dict
        Parameters
        ----------
        value: int
            Value of string in dict
        Returns
        -------
            word
        """
        return self.rev_dict[value]

    def encode(self, word: str, threshold_min: int = 0, threshold_max: int = None, oov: object = 0) -> object:
        """
        Encode word to int using dict, if word is out of vocabulary return 'oov'
        Parameters
        ----------
        word: str
            word to encode
        threshold_max: int
            maximum number of occurrences of word in dictionary, if above 'oov' is returned
        threshold_min: int
            minimum number of occurrences of word in dictionary, if under 'oov' is returned
        oov: object
            object to return of 'word' is out of vocabulary
        Returns
        -------
            Code of word or 'oov' value
        """
        try:
            if threshold_max:
                if self.occurrences[word] > threshold_max:
                    return oov
            if self.occurrences[word] < threshold_min:
                return oov
            return self.word_dict[word]
        except KeyError:
            return oov

    def encode_final(self, word: str, oov: object = 0) -> object:
        """ Encode word with final dictionary"""
        if self.final_dict is None:
            raise AssertionError("You need to run 'compile' method before using final dict")
        try:
            return self.final_dict[word]
        except KeyError:
            return oov

    def is_oov(self, word: str) -> bool:
        """
        Check if word is in dict
        Parameters
        ----------
        word: str
            word
        Returns
        -------
            True or False if word is in out of vocabulary
        """
        return False if word in self.word_dict else True

    def len_between(self, threshold_min: int = 0, threshold_max: int = None) -> int:
        """
        Count elements of dict that are between occurrences thresholds
        Parameters
        ----------
        threshold_min: int
            minimal number of occurrences
        threshold_max: int
            maximal number of occurrences

        Returns
        -------

        """
        cnt = 0
        for _, occ in self.occurrences.items():
            if threshold_max is None:
                if threshold_min <= occ:
                    cnt += 1
            else:
                if threshold_min <= occ <= threshold_max:
                    cnt += 1
        return cnt

    def compile(self, min_threshold: int = 0, max_threshold: int = None) -> dict:
        """ Apply codes to words in dictionary based on occurrences with ascending order"""
        final_dict = {}
        code = 0
        for word, occ in sorted(self.occurrences.items(), key=lambda x: x[1]):
            if max_threshold:
                if min_threshold <= occ <= max_threshold:
                    final_dict[word] = code
                    code += 1
            else:
                if min_threshold <= occ:
                    final_dict[word] = code
                    code += 1
        self.final_dict = final_dict
        return final_dict
