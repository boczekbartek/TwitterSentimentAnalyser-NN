import pytest
from coding import Coding


@pytest.fixture()
def coder():
    return Coding()


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


@pytest.mark.parametrize('current_dict, word, new_dict, expected_index', [
    ({'hello': 1, 'go': 2}, 'cat', {'hello': 1, 'go': 2, 'cat': 3}, 3),
    ({'hello': 1, 'go': 2}, 'go', {'hello': 1, 'go': 2}, 2)
])
def test_updating_dict(current_dict, word, new_dict, expected_index, coder):
    coder.word_dict = current_dict
    coder.max_code = max(current_dict.values())
    assert expected_index == coder.update_dict(word)
    assert coder.word_dict == new_dict


@pytest.mark.parametrize('current_dict, current_rev_dict, word, new_rev_dict, expected_index', [
    ({'hello': 1, 'go': 2}, {1: 'hello', 2: 'go'}, 'cat', {1: 'hello', 2: 'go', 3: 'cat'}, 3),
    ({'hello': 1, 'go': 2}, {1: 'hello', 2: 'go'}, 'go', {1: 'hello', 2: 'go'}, 2)
])
def test_updating_rev_dict(current_dict, current_rev_dict, word, new_rev_dict, expected_index, coder):
    coder.rev_dict = current_rev_dict
    coder.word_dict = current_dict
    coder.max_code = max(current_rev_dict.keys())
    assert expected_index == coder.update_dict(word)
    assert coder.rev_dict == new_rev_dict


@pytest.mark.parametrize('dictionary, word, expected_code', [
    ({'hello': 1, 'go': 2}, 'hello', 1),
    ({'hello': 1, 'go': 2}, 'cat', 0)
])
def test_encoding(dictionary, word, expected_code, coder):
    coder.word_dict = dictionary
    assert expected_code == coder.encode(word)


@pytest.mark.parametrize('reversed_dictionary, code, expected_word', [
    ({1: 'hello', 2: 'go'}, 1, 'hello')
])
def test_decoding(reversed_dictionary, code, expected_word, coder):
    coder.rev_dict = reversed_dictionary
    assert expected_word == coder.decode(code)


@pytest.mark.parametrize('reversed_dictionary, non_existing_code', [
    ({1: 'hello', 2: 'go'}, 3)
])
def test_decoding_raises_key_error(reversed_dictionary, non_existing_code, coder):
    coder.rev_dict = reversed_dictionary
    with pytest.raises(KeyError):
        coder.decode(non_existing_code)
