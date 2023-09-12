from imtmobileapps.math_util import top_three

my_nums = [1, 4328, 6, 8, 0, 3, 5, 6, 8, 92]


def test_result():
    assert top_three(my_nums) == [4328, 92, 8]
