import numpy as np
import pytest
from collections import defaultdict
from itertools import combinations
from customCV.brute import RSUKFold, RSUKFoldWithTolerance
from scipy.stats import chisquare


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_initialization(CVClass):
    # Test initialization
    cv = CVClass(n_splits=4, n_repeats=3, random_state=42)
    assert cv.n_splits == 4
    assert cv.n_repeats == 3
    assert cv.random_state == 42


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_split_functionality(CVClass):
    n_samples = 8
    n_repeats = 5
    n_splits = 2
    n_classes = 4
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
    X = np.arange(n_samples)
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


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_stratification_perfect(CVClass):
    CVClass = RSUKFold
    n_samples = 40
    n_repeats = 5
    n_splits = 10
    n_classes = 4
    y = np.arange(n_classes).repeat(n_samples // n_classes)
    X = np.arange(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    overall_ratios = np.bincount(y) / n_samples
    u, c = np.unique(y, return_counts=True)
    splits = list(cv.split(X, y))
    assert len(u) == n_classes
    assert (c[0] == c).all()
    assert len(splits) == n_repeats * n_splits
    for _, test_index in splits:
        test_y = y[test_index]
        fold_ratios = np.bincount(test_y) / len(test_index)
        assert len(test_index) == n_samples / n_splits
        for class_label in range(len(overall_ratios)):
            label_ratio = round(overall_ratios[class_label] * len(test_index))
            exp_ratio = fold_ratios[class_label] * len(test_index)
            assert abs(label_ratio - exp_ratio) <= 0


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_stratification_uneven(CVClass):
    n_samples = 39
    n_repeats = 5
    n_splits = 10
    n_classes = 4
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
    X = np.arange(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    overall_ratios = np.bincount(y) / n_samples
    u, c = np.unique(y, return_counts=True)
    splits = list(cv.split(X, y))
    assert len(u) == n_classes
    assert len(splits) == n_repeats * n_splits
    for _, test_index in splits:
        test_y = y[test_index]
        fold_ratios = np.bincount(test_y, minlength=n_classes) / len(test_index)
        for class_label in range(len(overall_ratios)):
            label_ratio = round(overall_ratios[class_label] * len(test_index))
            exp_ratio = fold_ratios[class_label] * len(test_index)
            assert abs(label_ratio - exp_ratio) <= 1


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_unique_folds_across_repeats(CVClass):
    n_samples = 40
    n_repeats = 5
    n_splits = 10
    n_classes = 4
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
    X = np.arange(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    seen_folds = set()
    for train_index, test_index in cv.split(X, y):
        fold = tuple(sorted(test_index))
        assert fold not in seen_folds
        seen_folds.add(fold)


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_random_state(CVClass):
    n_samples = 8
    n_classes = 4
    X = np.arange(n_samples)
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
    cv_0 = CVClass(n_splits=2, n_repeats=3, random_state=0)
    cv_00 = CVClass(n_splits=2, n_repeats=3, random_state=0)
    cv_1 = CVClass(n_splits=2, n_repeats=3, random_state=1)
    splits_0 = list(cv_0.split(X, y))
    splits_1 = list(cv_1.split(X, y))

    # Check that the splits are different due to different random states
    split_equal = [
        np.array_equal(splits_0[i][0], splits_1[i][0])
        and np.array_equal(splits_0[i][1], splits_1[i][1])
        for i in range(len(splits_0))
    ]
    assert not all(split_equal)

    # Check that the splits are the same when using the same random state
    splits_00 = list(cv_00.split(X, y))
    split_equal = [
        np.array_equal(splits_0[i][0], splits_00[i][0])
        and np.array_equal(splits_0[i][1], splits_00[i][1])
        for i in range(len(splits_0))
    ]
    assert all(split_equal)


def _test_exhaustion(CVClass, random_state):
    n_samples = 10
    n_splits = 5
    n_repeats = 9
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    seen_folds = set()
    for split_ix, (train_index, test_index) in enumerate(cv.split(X, y)):
        assert len(np.intersect1d(train_index, test_index)) == 0
        assert len(np.union1d(train_index, test_index)) == n_samples
        fold = tuple(sorted(test_index))
        assert fold not in seen_folds
        seen_folds.add(fold)

    split_size = n_samples // n_splits
    for comb in combinations(range(n_samples), split_size):
        assert tuple(sorted(comb)) in seen_folds


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_exhaustion_random(CVClass):
    _test_exhaustion(CVClass, random_state=42)


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_error_handling(CVClass):
    n_samples = 8
    n_classes = 4
    X = np.arange(n_samples)
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)

    # random_state must be an integer
    with pytest.raises(ValueError):
        cv = CVClass(n_splits=5, n_repeats=6, random_state=None)

    # Test with too many splits for the number of samples
    with pytest.raises(ValueError):
        cv = CVClass(n_splits=5, n_repeats=6)
        list(cv.split(X, y))

    # This error condition doesn't directly translate since we're not using groups,
    # but we can still test for an invalid configuration
    with pytest.raises(ValueError):
        cv = CVClass(n_splits=10, n_repeats=10)  # More splits than samples
        list(cv.split(X, y))


@pytest.mark.parametrize("CVClass", [RSUKFold])
def test_correlation(CVClass):
    """Test that there is no correlation between samples appearing in the same test set, i.e. sample_x is not more likely to appear with sample_y than any other sample."""

    n_samples = 24
    n_classes = 4
    n_splits = 3
    n_repeats = 80
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
    X = np.arange(n_samples)
    cv = CVClass(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    count_mat = np.zeros((n_samples, n_samples), dtype=int)
    for _, test_index in cv.split(X, y):
        # Increment counts for all pairs in test_index
        pairs = combinations(test_index, 2)
        for pair in pairs:
            pair = tuple(sorted(pair))
            count_mat[pair] += 1

    # select all samples from different classes
    n_same = n_samples // n_classes
    n_diff = n_samples - n_same
    count_mat_diff = np.ones((n_samples, n_samples), dtype=int) * -1
    for i in range(n_samples):
        label_i = y[i]
        for j in range(i + 1, n_samples):
            label_j = y[j]
            if label_i != label_j:
                pair = tuple(sorted((i, j)))
                count_mat_diff[pair] = count_mat[pair]

    print(count_mat)
    print(count_mat_diff)
    count_mat_diff_values = count_mat_diff[count_mat_diff != -1]
    stat, p = chisquare(count_mat_diff_values)
    print(f"Chi-square test p-value: {p:.2f}")
    assert p > 0.5, f"Chi-square test failed. p-value: {p:.2f}"

    n_diff_per_split = n_diff // n_splits
    expected_cooccurrences_diff = n_diff_per_split * n_repeats / n_diff

    # 1. Quantile Check (Focus on High Co-occurrences)
    cooccurrences_diff = count_mat_diff_values.flatten()
    quantile_levels = [0.05, 0.95]
    quantiles = np.quantile(cooccurrences_diff, quantile_levels)
    tolerance_factor = 1.75
    tolerance_abs = (tolerance_factor - 1) * expected_cooccurrences_diff
    low_quantile, high_quantile = quantiles
    assert low_quantile >= expected_cooccurrences_diff - tolerance_abs
    assert high_quantile <= expected_cooccurrences_diff + tolerance_abs

    # 2. Proportion Exceeding Threshold
    exceed_factor = 1.75
    num_higher = np.sum(
        cooccurrences_diff > exceed_factor * expected_cooccurrences_diff
    )
    num_lower = np.sum(
        cooccurrences_diff < (2 - exceed_factor) * expected_cooccurrences_diff
    )
    proportion_exceeding = num_higher / len(cooccurrences_diff)
    proportion_lower = num_lower / len(cooccurrences_diff)
    max_proportion_exceeding = 0.01
    assert (
        proportion_exceeding <= max_proportion_exceeding
    ), f"Proportion exceeding threshold ({proportion_exceeding:.2f}) is too high. Possible correlation."
    assert (
        proportion_lower <= max_proportion_exceeding
    ), f"Proportion exceeding threshold ({proportion_lower:.2f}) is too high. Possible correlation."

    # 3. Maximum Co-occurrence Check (Extreme Values)
    max_cooccurrences_diff = np.max(cooccurrences_diff)
    max_allowed_cooccurrences_diff = int(3 * expected_cooccurrences_diff)  # adjust
    assert (
        max_cooccurrences_diff < max_allowed_cooccurrences_diff
    ), f"Maximum co-occurrences ({max_cooccurrences_diff}) is too high. Possible correlation."


def test_tolerance():
    n_samples = 8
    n_repeats = 5
    n_splits = 4
    n_classes = 2
    y = (np.arange(n_samples) // np.ceil(n_samples / n_classes)).astype(int)
    X = np.arange(n_samples)

    # Fails with no tolerance
    cv_no_tol = RSUKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    with pytest.raises(ValueError):
        list(cv_no_tol.split(X, y))

    tolerance = 1
    cv_tol = RSUKFoldWithTolerance(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42, tolerance=tolerance
    )
    test_indices_count = defaultdict(int)
    for split_ix, (train_index, test_index) in enumerate(cv_tol.split(X, y)):
        assert len(np.intersect1d(train_index, test_index)) == 0
        assert len(np.union1d(train_index, test_index)) == len(X)
        for idx in test_index:
            test_indices_count[idx] += 1
    for count in test_indices_count.values():
        assert count == n_repeats
