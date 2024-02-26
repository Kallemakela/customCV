import numpy as np
import random
from itertools import combinations
from scipy.special import comb

def unique_random_combinations(iterable, r, rng):
    """Generate unique random combinations."""
    pool = tuple(iterable)
    n = len(pool)
    indices = list(range(n))
    rng.shuffle(indices)
    for i in combinations(indices, r):
        yield tuple(sorted(pool[j] for j in i))

class RepeatedUniqueFoldKFold:
    """
    Repeated KFold with unique folds across all repeats.
    Each sample appears in the test set exactly once per repeat.
    All folds are unique, even across repeats.
    Parameters:
    n_splits (int): Number of folds. Must be at least 2.
    n_repeats (int): Number of times cross-validator needs to be repeated.
    random_state (int, RandomState instance or None, optional): random seed

    The fold search is performed in a depth-first manner, by first finding a unique fold from groups that have not yet been used on this repeat, adding it to a list of potential folds, and moving on to the next fold.
    If the fold search ends up in a situation where it is not possible to generate valid folds from the remaining groups, it backtracks by one fold and marks the backtracked fold as exhausted.
    """
    def __init__(self, n_splits=5, n_repeats=10, random_state=42, max_iter=int(1e6)):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)

    def comb_generator(self, iterable, r):
        return unique_random_combinations(iterable, r, self.rng)

    def find_next_fold_(self, available_samples, fold_size, used_folds, current_fold_path, exhausted_paths):
        fold_gen = self.comb_generator(available_samples, fold_size)
        while True:
            try:
                fold = tuple(sorted(next(fold_gen)))
                potential_path = tuple(current_fold_path + [fold])
                if fold not in used_folds and potential_path not in exhausted_paths:
                    return fold
            except StopIteration:
                return None

    def find_next_folds_(self, n_samples, samples_per_fold, used_folds):
        current_fold_path = []
        available_samples = set(range(n_samples))
        exhausted_paths = set()
        i = 0
        while len(current_fold_path) < self.n_splits:
            fold_ix = len(current_fold_path)
            fold_size = samples_per_fold[fold_ix]
            next_fold = self.find_next_fold_(available_samples, fold_size, used_folds, current_fold_path, exhausted_paths)

            if next_fold is None:
                if len(current_fold_path) == 0:
                    raise ValueError("Not enough samples to create unique folds.")

                exhausted_paths.add(tuple(current_fold_path))
                exhausted_fold = current_fold_path.pop()
                available_samples.update(exhausted_fold)
                i += 1
                if i > self.max_iter:
                    raise ValueError("Max iterations reached.")
            else:
                current_fold_path.append(next_fold)
                available_samples -= set(next_fold)
        
        used_folds.update(current_fold_path)
        return current_fold_path

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        if n_samples < self.n_splits:
            raise ValueError("Number of samples must be at least equal to n_splits.")
        
        # Calculate the number of samples per fold
        samples_per_fold = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        samples_per_fold[:n_samples % self.n_splits] += 1

        used_folds = set()
        for ri in range(self.n_repeats):
            folds = self.find_next_folds_(n_samples, samples_per_fold, used_folds)
            for fold in folds:
                test_idx = np.array(fold)
                train_idx = np.array([i for i in range(n_samples) if i not in fold])
                yield train_idx, test_idx

class RepeatedUniqueFoldKFoldPG(RepeatedUniqueFoldKFold):
    def __init__(self, n_splits=5, n_repeats=10, random_state=42, max_iter=int(1e6), verbose=0):
        super().__init__(n_splits, n_repeats, random_state, max_iter)
        self.pre_generated_folds = []
        self.max_resets = 10
        self.verbose = verbose

    def pre_generate_folds(self, n_samples):
        """
        Pre-generates all the folds for the repeats and stores them.
        If it fails to generate unique folds, it resets and tries again from the beginning.
        """
        samples_per_fold = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        samples_per_fold[:n_samples % self.n_splits] += 1

        used_folds = set()
        attempt = 0

        while len(self.pre_generated_folds) < self.n_splits * self.n_repeats:
            try:
                folds = self.find_next_folds_(n_samples, samples_per_fold, used_folds)
                self.pre_generated_folds.extend(folds)
            except ValueError as e:
                # Reset everything and try again if unique folds cannot be generated
                if str(e) == "Not enough samples to create unique folds.":
                    self.pre_generated_folds.clear()
                    used_folds.clear()
                    attempt += 1
                    if self.verbose > 0:
                        print(f"Unique fold generation reached a dead end. Resetting and trying again. Attempt {attempt}/{self.max_resets}")
                    if attempt >= self.max_resets:
                        raise ValueError("max_resets reached. Cannot generate unique folds.")
                else:
                    raise  # If the error is not about unique fold generation, re-raise it


    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        if n_samples < self.n_splits:
            raise ValueError("Number of samples must be at least equal to n_splits.")

        # Pre-generate folds if not already done
        if not self.pre_generated_folds:
            self.pre_generate_folds(n_samples)

        # Yield pre-generated fold indices
        for fold in self.pre_generated_folds:
            test_idx = np.array(fold)
            train_idx = np.array([i for i in range(n_samples) if i not in fold])
            yield train_idx, test_idx
