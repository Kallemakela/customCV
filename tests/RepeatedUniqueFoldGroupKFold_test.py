import numpy as np
import pytest
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.model_selection import check_cv
from customCV.group import RepeatedUniqueFoldGroupKFold


def test_initialization():
    # Test initialization
    cv = RepeatedUniqueFoldGroupKFold(n_splits=4, n_repeats=3)
    assert cv.n_splits == 4
    assert cv.n_repeats == 3
    assert cv.random_state == None

def test_split_functionality():
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    n_groups = len(np.unique(groups))
    n_repeats = 3
    n_splits = 2
    n_folds_per_repeat = n_groups // n_splits
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=n_splits, n_repeats=n_repeats)

    test_groups = defaultdict(lambda: defaultdict(int))
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        
        # Check that the train and test groups are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == len(groups)

        # check that each group is exactly once in the test set per repeat
        split_repeat_ix = split_ix // n_folds_per_repeat
        fold_test_groups = np.unique(groups[test_index])
        for group in fold_test_groups:
            test_groups[split_repeat_ix][group] += 1
        
    for repeat_ix in range(n_repeats):
        for group in range(1, n_groups + 1):
            assert test_groups[repeat_ix][group] == 1    
        
def test_unique_folds_across_repeats():
    groups = np.arange(8) // 2
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=3)

    seen_folds = set()
    for train_index, test_index in cv.split(X, y, groups):
        fold = tuple(sorted(groups[test_index]))
        assert fold not in seen_folds
        seen_folds.add(fold)

def test_random_state():
    groups = np.arange(8) // 2
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv_0 = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=3, random_state=0)
    cv_1 = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=3, random_state=1)
    splits_0 = list(cv_0.split(X, y, groups))
    splits_1 = list(cv_1.split(X, y, groups))

    # Check that the splits are different
    split_equal = [np.array_equal(splits_0[i][0], splits_1[i][0]) for i in range(len(splits_0))]
    assert not all(split_equal)

def test_exhaustion():
    fold_size = 2
    groups = np.arange(40) // fold_size
    n_groups = len(np.unique(groups))
    n_splits = n_groups // fold_size
    n_repeats = 3
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=n_splits, n_repeats=n_repeats)

    seen_folds = set()
    test_groups = defaultdict(lambda: defaultdict(int))
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        
        # Check that the train and test groups are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == len(groups)

        fold = tuple(sorted(groups[test_index]))
        print(fold)
        assert fold not in seen_folds
        seen_folds.add(fold)

        # check that each group is exactly once in the test set per repeat
        split_repeat_ix = split_ix // n_splits
        fold_test_groups = np.unique(groups[test_index])
        for group in fold_test_groups:
            test_groups[split_repeat_ix][group] += 1

    for repeat_ix in range(n_repeats):
        for group in range(n_groups):
            assert test_groups[repeat_ix][group] == 1   

def test_exhaustion_random():
    fold_size = 2
    groups = np.arange(40) // fold_size
    n_groups = len(np.unique(groups))
    n_splits = n_groups // fold_size
    n_repeats = 3
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    seen_folds = set()
    test_groups = defaultdict(lambda: defaultdict(int))
    print('\n')
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        
        # Check that the train and test groups are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == len(groups)

        fold = tuple(sorted(groups[test_index]))
        print(fold)
        assert fold not in seen_folds
        seen_folds.add(fold)

        # check that each group is exactly once in the test set per repeat
        split_repeat_ix = split_ix // n_splits
        fold_test_groups = np.unique(groups[test_index])
        for group in fold_test_groups:
            test_groups[split_repeat_ix][group] += 1

    for repeat_ix in range(n_repeats):
        for group in range(n_groups):
            assert test_groups[repeat_ix][group] == 1   

def test_uneven():
    groups = np.array([0,0,0,1,1,2,2,3,3])
    n_groups = len(np.unique(groups))
    n_splits = 2
    n_repeats = 3
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=n_splits, n_repeats=n_repeats)

    seen_folds = set()
    test_groups = defaultdict(lambda: defaultdict(int))
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        
        # Check that the train and test groups are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == len(groups)

        fold = tuple(sorted(groups[test_index]))
        print(fold)
        assert fold not in seen_folds
        seen_folds.add(fold)

        # check that each group is exactly once in the test set per repeat
        split_repeat_ix = split_ix // n_splits
        fold_test_groups = np.unique(groups[test_index])
        for group in fold_test_groups:
            test_groups[split_repeat_ix][group] += 1

    for repeat_ix in range(n_repeats):
        for group in range(n_groups):
            assert test_groups[repeat_ix][group] == 1   


def test_error_handling():
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    
    # Test with too many splits for the number of groups
    with pytest.raises(ValueError):
        cv = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=4)
        list(cv.split(X, y, groups))

    # Test with not enough unique groups for the required number of folds
    with pytest.raises(ValueError):
        cv = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=10)
        list(cv.split(X, y, groups))
