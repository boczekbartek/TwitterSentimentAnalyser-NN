class Coding:
    def __init__(self):
        self.word_dict = dict()
        self.rev_dict = dict()
        self.max_code = 0

    def make_dict(self, words: list) -> dict:
        """
        Make dict from list of words, set to self.word_dict. There are
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
        if word not in self.word_dict:
            new_code = self.max_code + 1
            self.word_dict[word] = new_code
            self.rev_dict[new_code] = word
            self.max_code = new_code
        return self.word_dict[word]

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

    def encode(self, word: str)-> int:
        """
        Encode word to int using dict, if word is out of vocabulary return 0
        Parameters
        ----------
        word: str
            word to encode
        Returns
        -------
            Code of word
        """
        try:
            return self.word_dict[word]
        except KeyError:
            return 0

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
