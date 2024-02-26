import numpy as np
import pytest
from collections import defaultdict
from itertools import combinations
from scipy.stats import chisquare
from customCV.repeated import RepeatedUniqueFoldKFoldPG, RepeatedUniqueFoldKFold

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_initialization(CVClass):
    # Test initialization
    cv = CVClass(n_splits=4, n_repeats=3, random_state=42)
    assert cv.n_splits == 4
    assert cv.n_repeats == 3
    assert cv.random_state == 42

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_split_functionality(CVClass):
    n_samples = 10
    n_repeats = 3
    n_splits = 2
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    test_indices_count = defaultdict(int)
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y)):
        
        # Check that the train and test indices are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == len(X)

        # Count how often each index appears in the test sets
        for idx in test_index:
            test_indices_count[idx] += 1
        
    # Each sample should appear in the test set exactly n_repeats times
    for count in test_indices_count.values():
        assert count == n_repeats

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_unique_folds_across_repeats(CVClass):
    n_samples = 8
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=2, n_repeats=3, random_state=42)

    seen_folds = set()
    for train_index, test_index in cv.split(X, y):
        fold = tuple(sorted(test_index))
        assert fold not in seen_folds
        seen_folds.add(fold)

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_random_state(CVClass):
    n_samples = 8
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv_0 = CVClass(n_splits=2, n_repeats=3, random_state=0)
    cv_1 = CVClass(n_splits=2, n_repeats=3, random_state=1)
    splits_0 = list(cv_0.split(X, y))
    splits_1 = list(cv_1.split(X, y))

    # Check that the splits are different due to different random states
    split_equal = [np.array_equal(splits_0[i][0], splits_1[i][0]) and np.array_equal(splits_0[i][1], splits_1[i][1]) for i in range(len(splits_0))]
    assert not all(split_equal)

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_exhaustion(CVClass):
    n_samples = 40
    n_splits = 4
    n_repeats = 3
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

    seen_folds = set()
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y)):
        
        # Check that the train and test indices are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == n_samples

        fold = tuple(sorted(test_index))
        assert fold not in seen_folds
        seen_folds.add(fold)

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_exhaustion_random(CVClass):
    n_samples = 40
    n_splits = 4
    n_repeats = 3
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    seen_folds = set()
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y)):
        
        # Check that the train and test indices are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == n_samples

        fold = tuple(sorted(test_index))
        assert fold not in seen_folds
        seen_folds.add(fold)

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_uneven(CVClass):
    n_samples = 9  # Uneven number of samples
    n_splits = 2
    n_repeats = 3
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    seen_folds = set()
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y)):
        
        # Check that the train and test indices are disjoint
        assert len(np.intersect1d(train_index, test_index)) == 0
        # check that all samples are included in either train or test
        assert len(np.union1d(train_index, test_index)) == n_samples

        fold = tuple(sorted(test_index))
        assert fold not in seen_folds
        seen_folds.add(fold)

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_error_handling(CVClass):
    n_samples = 8
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    
    # Test with too many splits for the number of samples
    with pytest.raises(ValueError):
        cv = CVClass(n_splits=5, n_repeats=6)
        list(cv.split(X, y))

    # This error condition doesn't directly translate since we're not using groups,
    # but we can still test for an invalid configuration
    with pytest.raises(ValueError):
        cv = CVClass(n_splits=10, n_repeats=10)  # More splits than samples
        list(cv.split(X, y))

@pytest.mark.parametrize("CVClass", [RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG])
def test_correlation(CVClass):
    n_samples = 12
    n_splits = 3
    n_repeats = 80
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # Initialize a matrix to count occurrences of each sample pair in the same test set
    count_mat = np.zeros((n_samples, n_samples), dtype=int)
    for _, test_index in cv.split(X, y):
        # Increment counts for all pairs in test_index
        pairs = combinations(test_index, 2)
        for pair in pairs:
            pair = tuple(sorted(pair))
            count_mat[pair] += 1

    stat, p = chi_square_test_upper_triangular(count_mat)
    print(count_mat)
    print(stat, p)
    assert p > .5

def chi_square_test_upper_triangular(M):
    """
    Performs a chi-square goodness-of-fit test on the counts of sample pairs
    from an upper triangular matrix, where M[i, j] represents the count of 
    times sample i and sample j have appeared together.
    """
    observed_frequencies = M[np.triu_indices_from(M, k=1)]
    total_appearances = observed_frequencies.sum()
    num_pairs = len(observed_frequencies)
    expected_frequency = total_appearances / num_pairs if num_pairs > 0 else 0

    chi2_stat, p_value = chisquare(observed_frequencies, [expected_frequency] * num_pairs) if num_pairs > 0 else (np.nan, np.nan)
    
    return chi2_stat, p_value