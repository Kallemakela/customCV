import numpy as np
import pytest
from customCV.group import GroupCVWrapper
from sklearn.model_selection import KFold, RepeatedKFold


def test_kfold():
    groups = np.array([0, 0, 1, 0])
    kfold = KFold(n_splits=2)
    cv = GroupCVWrapper(kfold)
    for train, test in cv.split(groups=groups):
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        assert len(test_groups) == 1
        assert len(train_groups) == 1
        assert test_groups.isdisjoint(train_groups)


def test_split():
    groups = [0, 0, 1, 0, 2, 1]
    base_cv = KFold(n_splits=2, shuffle=True, random_state=42)
    wrapper = GroupCVWrapper(base_cv)
    for train_idx, test_idx in wrapper.split(groups=groups):
        # For each group, samples must be exclusively in train or test.
        for group in np.unique(groups):
            indices = np.where(np.array(groups) == group)[0]
            # Equation: set(indices) ∩ set(train_idx) == ∅ or set(indices) ∩ set(test_idx) == ∅
            assert not (set(indices) & set(train_idx) and set(indices) & set(test_idx))


def test_get_n_splits():

    def assert_match(wrapper, groups):
        split_counter = 0
        for _ in wrapper.split(groups=groups):
            split_counter += 1
        assert split_counter == wrapper.get_n_splits(groups=groups)

    np.random.seed(42)
    groups = np.random.randint(0, 10, 100)
    wrapper = GroupCVWrapper(KFold(n_splits=10, shuffle=True, random_state=42))
    assert_match(wrapper, groups)

    groups = np.random.randint(0, 10, 100)
    wrapper = GroupCVWrapper(RepeatedKFold(n_splits=10, n_repeats=5, random_state=42))
    assert_match(wrapper, groups)
