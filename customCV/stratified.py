from .repeated import RepeatedUniqueFoldKFold
from collections import Counter
import numpy as np

from .stratified_generator import stratified_combinations
from .utils import get_allocation


class RepeatedStratifiedUniqueFoldKFold(RepeatedUniqueFoldKFold):
    """
    Stratified Repeated KFold with unique folds across all repeats.
    Extends RepeatedUniqueFoldKFold to include stratification based on 'y'.

    Very slow. Can be optimized if needed.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=2
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `split`.
    max_iter : int, default=1e6
        Maximum number of iterations to attempt to find unique folds.
    tolerance : int, default=0
        Maximum allowable deviation from expected class distribution.
    """

    def __init__(
        self,
        n_splits=5,
        n_repeats=2,
        random_state=42,
        max_iter=int(1e6),
        tolerance=0,
        max_tries_per_fold=100,
        verbose=0,
        warn=True,
    ):
        super().__init__(n_splits, n_repeats, random_state, max_iter, verbose, warn)
        self.tolerance = tolerance
        self.max_tries_per_fold = max_tries_per_fold

        if warn:
            print(
                f"Warning: Not properly tested. Use at your own risk. Check splits before using them."
            )

    def comb_generator(self, iterable, r):
        return stratified_combinations(iterable, r, self.rng)

    def _is_fold_stratified(self, fold_with_labels, overall_ratios, fold_size):
        """Checks if a fold is stratified based on class distribution."""
        fold_class_counts = Counter(label for _, label in fold_with_labels)
        for (
            class_label,
            expected_ratio,
        ) in overall_ratios.items():
            count = fold_class_counts.get(class_label, 0)
            expected_count = round(expected_ratio * fold_size)
            if abs(count - expected_count) > self.tolerance:
                return False
        return True

    def get_allocation(self, y):
        return get_allocation(y, self.n_splits)

    def find_next_fold_(
        self,
        available_samples_by_class,
        fold_size,
        used_folds,
        current_fold_path,
        exhausted_paths,
        class_counts,
        overall_ratios,
    ):
        """
        Finds the next unique fold, considering stratification.  Overrides the base class method.
        """
        # Build a list of all available samples, including class labels
        available_samples = []
        for class_label, samples in available_samples_by_class.items():
            available_samples.extend([(sample, class_label) for sample in samples])

        fold_gen = self.comb_generator(available_samples, fold_size)

        fold_tries = 0
        while True:
            try:
                fold_with_labels = next(fold_gen)
                fold = tuple(
                    sorted(sample for sample, _ in fold_with_labels)
                )  # Extract sample indices
                potential_path = tuple(current_fold_path + [fold])

                if (
                    fold not in used_folds
                    and potential_path not in exhausted_paths
                    and self._is_fold_stratified(
                        fold_with_labels, overall_ratios, fold_size
                    )
                ):
                    return fold
                fold_tries += 1
                if fold_tries > self.max_tries_per_fold:
                    return None
            except StopIteration:
                return None

    def find_next_folds_(self, n_samples, samples_per_fold, used_folds, y):
        """
        Finds the next set of unique folds, considering stratification.  Overrides the base class method.
        """
        current_fold_path = []
        available_samples_by_class = {}
        for i in range(n_samples):
            class_label = y[i]
            if class_label not in available_samples_by_class:
                available_samples_by_class[class_label] = []
            available_samples_by_class[class_label].append(i)

        class_counts = {
            class_label: len(samples)
            for class_label, samples in available_samples_by_class.items()
        }
        overall_ratios = {
            class_label: count / n_samples
            for class_label, count in class_counts.items()
        }

        exhausted_paths = set()
        i = 0
        while len(current_fold_path) < self.n_splits:
            fold_ix = len(current_fold_path)
            fold_size = samples_per_fold[fold_ix]
            next_fold = self.find_next_fold_(
                available_samples_by_class,
                fold_size,
                used_folds,
                current_fold_path,
                exhausted_paths,
                class_counts,
                overall_ratios,
            )

            if next_fold is None:
                if len(current_fold_path) == 0:
                    raise ValueError(
                        "Could not find a valid folds. There may not be enough samples to create unique, stratified folds. You can try 1) reducing total number of splits, 2) increasing max_tries_per_fold, 3) increasing max_iter, or 4) increasing tolerance."
                    )

                exhausted_paths.add(tuple(current_fold_path))
                exhausted_fold = current_fold_path.pop()
                # Return samples to available pool, by class
                for sample_index in exhausted_fold:
                    class_label = y[sample_index]
                    available_samples_by_class[class_label].append(sample_index)

                i += 1
                if i > self.max_iter:
                    raise ValueError("Max iterations reached.")
            else:
                current_fold_path.append(next_fold)
                # Remove selected samples from available pool, by class
                for sample_index in next_fold:
                    class_label = y[sample_index]
                    available_samples_by_class[class_label].remove(sample_index)

        used_folds.update(current_fold_path)
        return current_fold_path

    def split(self, X=None, y=None, groups=None):
        """
        Generate indices to split data, stratified by 'y'. Overrides the base class method.
        """
        if y is None:
            raise ValueError(
                "The 'y' parameter (target variable) is required for stratification."
            )
        n_samples = y.shape[0]
        if n_samples < self.n_splits:
            raise ValueError("Number of samples must be at least equal to n_splits.")

        samples_per_fold = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        samples_per_fold[: n_samples % self.n_splits] += 1

        used_folds = set()
        for ri in range(self.n_repeats):
            folds = self.find_next_folds_(n_samples, samples_per_fold, used_folds, y)
            for fold in folds:
                test_idx = np.array(fold)
                train_idx = np.array([i for i in range(n_samples) if i not in fold])
                yield train_idx, test_idx
