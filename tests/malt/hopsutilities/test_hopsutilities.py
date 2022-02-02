import malt.hopsutilities as hsutil


def test_hops_path_to_tuple():
    assert hsutil.hops_path_to_tuple("{0;0;0}") == (0, 0, 0)
    assert hsutil.hops_path_to_tuple("{0;1;0;5}") == (0, 1, 0, 5)
    assert hsutil.hops_path_to_tuple("{0;1}") == (0, 1)
    assert hsutil.hops_path_to_tuple("{0;0;1;2;3}") == (0, 0, 1, 2, 3)


def test_hops_paths_to_tuples():
    fake_paths = ["{0;0;0}", "{0;0;1}", "{0;0;2}", "{0;1;2;3;5}"]
    paths = hsutil.hops_paths_to_tuples(fake_paths)
    assert paths == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2, 3, 5)]


def test_hops_tree_verify():
    faketree = {"{0;1;2}": [0, 1, 2],
                "{0;0;0}": [0, 0, 0]}
    assert hsutil.hops_tree_verify(faketree) is True
    faketree = {"{0;1;2}": [0, 1, 2],
                "{0;0;0;1}": [0, 0, 0]}
    assert hsutil.hops_tree_verify(faketree) is False
