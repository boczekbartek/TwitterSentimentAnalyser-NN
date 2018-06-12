import pytest
from sentiment_analyser.process import pad_or_truncate


@pytest.mark.parametrize('in_list, target_len, end, pad_value, expected', [
    ([1, 2, 3, 4], 3, True, "", [1, 2, 3]),
    ([1, 2, 3, 4], 3, False, "", [2, 3, 4]),
    ([1, 2, 3, 4], 5, True, 0, [1, 2, 3, 4, 0]),
    ([1, 2, 3, 4], 5, False, 0, [0, 1, 2, 3, 4]),
    ([1, 2, 3, 4], 6, False, "0", ["0", "0", 1, 2, 3, 4])
])
def test_pad_or_truncate(in_list, target_len, end, pad_value, expected):
    assert expected == pad_or_truncate(in_list=in_list,
                                       target_len=target_len,
                                       end=end,
                                       pad_value=pad_value)
