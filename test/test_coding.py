import pytest
from coding import Coding


@pytest.fixture()
def coder():
    return Coding()


@pytest.fixture()
def dictionary():
    return {'hello': 1, 'go': 2}


@pytest.fixture()
def rev_dict():
    return {1: 'hello', 2: 'go'}


@pytest.mark.parametrize('words, expected', [
    (['hello', 'how', 'are', 'you', 'doing'], {'hello': 1, 'how': 2, 'are': 3, 'you': 4, 'doing': 5}),
    (['hello', 'hello', 'hello', 'go', 'go', 'go'], {'hello': 1, "go": 2})
])
def test_making_dictionary_straight(words, expected, coder):
    assert expected == coder.make_dict(words)


@pytest.mark.parametrize('words, expected', [
    (['hello', 'how', 'are', 'you', 'doing'], {1: 'hello', 2: 'how', 3: 'are', 4: 'you', 5: 'doing'}),
    (['hello', 'hello', 'hello', 'go', 'go', 'go'], {1: 'hello', 2: 'go'})
])
def test_making_dictionary_reversed(words, expected, coder):
    coder.make_dict(words)
    assert expected == coder.rev_dict


@pytest.mark.parametrize('word, new_dict, expected_index', [
    ('cat', {'hello': 1, 'go': 2, 'cat': 3}, 3),
    ('go', {'hello': 1, 'go': 2}, 2)
])
def test_updating_dict(dictionary, word, new_dict, expected_index, coder):
    coder.word_dict = dictionary
    coder.max_code = max(dictionary.values())
    assert expected_index == coder.update_dict(word)
    assert coder.word_dict == new_dict


@pytest.mark.parametrize('word, new_rev_dict, expected_index', [
    ('cat', {1: 'hello', 2: 'go', 3: 'cat'}, 3),
    ('go', {1: 'hello', 2: 'go'}, 2)
])
def test_updating_rev_dict(dictionary, rev_dict, word, new_rev_dict, expected_index, coder):
    coder.rev_dict = rev_dict
    coder.word_dict = dictionary
    coder.max_code = max(rev_dict.keys())
    assert expected_index == coder.update_dict(word)
    assert coder.rev_dict == new_rev_dict


@pytest.mark.parametrize('word, expected_code', [
    ('hello', 1),
    ('cat', 0)
])
def test_encoding(dictionary, word, expected_code, coder):
    coder.word_dict = dictionary
    assert expected_code == coder.encode(word)


@pytest.mark.parametrize('code, expected_word', [
    (1, 'hello')
])
def test_decoding(rev_dict, code, expected_word, coder):
    coder.rev_dict = rev_dict
    assert expected_word == coder.decode(code)


@pytest.mark.parametrize('non_existing_code', [
    tuple([3])
])
def test_decoding_raises_key_error(rev_dict, non_existing_code, coder):
    coder.rev_dict = rev_dict
    with pytest.raises(KeyError):
        coder.decode(non_existing_code)


@pytest.mark.parametrize('word, expected', [
    ('hello', False),
    ('cat', True),
    ('go', False)
])
def test_out_of_vocabulary(dictionary, word, expected, coder):
    coder.word_dict = dictionary
    assert expected == coder.is_oov(word)
