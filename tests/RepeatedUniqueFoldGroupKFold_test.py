import numpy as np
import pytest
from collections import defaultdict
from itertools import combinations
from scipy.stats import chisquare
from customCV.group import RepeatedUniqueFoldGroupKFoldPG, RepeatedUniqueFoldGroupKFold


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_initialization(CVClass):
    # Test initialization
    cv = RepeatedUniqueFoldGroupKFold(n_splits=4, n_repeats=3)
    assert cv.n_splits == 4
    assert cv.n_repeats == 3
    assert cv.random_state == 42


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_split_functionality(CVClass):
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


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_unique_folds_across_repeats(CVClass):
    groups = np.arange(8) // 2
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=3)

    seen_folds = set()
    for train_index, test_index in cv.split(X, y, groups):
        fold = tuple(sorted(groups[test_index]))
        assert fold not in seen_folds
        seen_folds.add(fold)


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_random_state(CVClass):
    groups = np.arange(8) // 2
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv_0 = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=3, random_state=0)
    cv_1 = RepeatedUniqueFoldGroupKFold(n_splits=2, n_repeats=3, random_state=1)
    splits_0 = list(cv_0.split(X, y, groups))
    splits_1 = list(cv_1.split(X, y, groups))

    # Check that the splits are different
    split_equal = [
        np.array_equal(splits_0[i][0], splits_1[i][0]) for i in range(len(splits_0))
    ]
    assert not all(split_equal)


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_exhaustion(CVClass):
    fold_size = 2
    groups = np.arange(40) // fold_size
    n_groups = len(np.unique(groups))
    n_splits = n_groups // fold_size
    n_repeats = 3
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=None
    )

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


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_exhaustion_random(CVClass):
    fold_size = 2
    groups = np.arange(40) // fold_size
    n_groups = len(np.unique(groups))
    n_splits = n_groups // fold_size
    n_repeats = 3
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=0
    )

    seen_folds = set()
    test_groups = defaultdict(lambda: defaultdict(int))
    print("\n")
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


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_uneven(CVClass):
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
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


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_error_handling(CVClass):
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


@pytest.mark.parametrize(
    "CVClass", [RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG]
)
def test_correlation(CVClass):
    """Test that there is no correlation between samples appearing in the same test set, i.e. sample_x is not more likely to appear with sample_y than any other sample."""
    fold_size = 5
    groups = np.arange(60) // fold_size
    n_groups = len(np.unique(groups))
    n_splits = n_groups // fold_size
    n_repeats = 50
    X = np.ones(len(groups))
    y = np.ones(len(groups))
    cv = RepeatedUniqueFoldGroupKFold(n_splits=n_splits, n_repeats=n_repeats)

    count_mat = np.zeros((n_groups, n_groups), dtype=int)
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        fold_test_groups = np.unique(groups[test_index])

        # each pair to count mat
        pairs = combinations(fold_test_groups, 2)
        for pair in pairs:
            count_mat[pair] += 1

    stat, p = chi_square_test_upper_triangular(count_mat)
    print(count_mat)
    print(stat, p)
    assert p > 0.5


def chi_square_test_upper_triangular(M):
    """
    Performs a chi-square goodness-of-fit test on the counts of element pairs
    from an upper triangular matrix, where M[i, j] represents the count of
    times element i and element j have appeared together.

    Args:
    - M (numpy.ndarray): An upper triangular matrix of pair appearance counts.

    Returns:
    - chi2_stat (float): The chi-square statistic.
    - p_value (float): The p-value of the test.
    """
    # Flatten the matrix to get the observed frequencies, ignoring zeros and the diagonal
    observed_frequencies = M[np.triu_indices_from(M, k=1)]

    # Calculate the expected frequency
    total_appearances = observed_frequencies.sum()
    num_pairs = len(observed_frequencies)
    expected_frequency = total_appearances / num_pairs if num_pairs > 0 else 0

    # Perform the chi-square goodness-of-fit test
    chi2_stat, p_value = (
        chisquare(observed_frequencies, [expected_frequency] * num_pairs)
        if num_pairs > 0
        else (np.nan, np.nan)
    )

    return chi2_stat, p_value
