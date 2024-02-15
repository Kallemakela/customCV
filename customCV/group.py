import numpy as np
import random
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from scipy.special import comb
from itertools import combinations

class GroupShuffleSplit_(GroupShuffleSplit):
    """GroupShuffleSplit that accepts the shuffle parameter in the constructor so that it can be used with _RepeatedSplits"""
    def __init__(self, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.shuffle = shuffle

class RepeatedGroupShuffleSplit(_RepeatedSplits):
    """
    Randomized group CV iterator with possible overlap. 
    """
    def __init__(self, **kwargs):
        super().__init__(GroupShuffleSplit_, **kwargs)

class RandomGroupKfold(GroupKFold):
    """GroupKFold that accepts the shuffle parameter. Ignores group sizes."""
    def __init__(self, shuffle=True, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.shuffle = shuffle
        self.random_state = random_state
        
        if isinstance(self.random_state, int):
            self.rng = np.random.default_rng(random_state)
        elif isinstance(self.random_state, np.random.RandomState):
            self.rng = self.random_state

    def _iter_test_indices(self, X, y, groups):
        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        group_to_fold = np.arange(n_groups) % self.n_splits
        if self.shuffle:
            group_to_fold = self.rng.permutation(group_to_fold)
        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

class RepeatedGroupKfold(_RepeatedSplits):
    """
    Repeated group CV iterator with no overlap (per repeat).
    """
    def __init__(self, **kwargs):
        super().__init__(RandomGroupKfold, **kwargs)

def unique_random_combinations(iterable, r, rng):
    """A generator that yields unique random combinations of length r from the iterable."""
    seen = set()
    items = list(iterable)
    total_combinations = comb(len(items), r, exact=True)
    
    while len(seen) < total_combinations:
        c = rng.sample(items, r)
        combination = tuple(sorted(c))
        hashed_combination = hash(combination)
        if hashed_combination not in seen:
            seen.add(hashed_combination)
            yield combination

class RepeatedUniqueFoldGroupKFold:
    """
    Repeated GroupKFold with unique folds across all repeats for even or uneven fold sizes.

    Each group appears in the test set exactly once per repeat.
    All folds are unique, even across repeats.
    Randomizaton works by generating a mapping from the original groups to a random permutation of the groups on each repeat.
    
    The fold search is performed in a depth-first manner, by first finding a unique fold from groups that have not yet been used on this repeat, adding it to a list of potential folds, and moving on to the next fold.
    If the fold search ends up in a situation where it is not possible to generate valid folds from the remaining groups, it backtracks by one fold and marks the backtracked fold as exhausted.
    
    Parameters:
    n_splits (int): Number of folds. Must be at least 2.
    n_repeats (int): Number of times cross-validator needs to be repeated.
    random_state (int, RandomState instance or None, optional): random seed
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=42, max_iter=int(1e6)):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.max_iter = max_iter

        if random_state == None:
            print(f"Warning: With random_state=None Some groups might be more likely to appear in the same split.")
            self.rng = None
        else:
            self.rng = random.Random(random_state)

    def comb_generator(self, iterable, r):
        if self.rng is not None:
            return unique_random_combinations(iterable, r, self.rng)
        else:
            return combinations(iterable, r)

    def find_next_fold_(self, available_groups, fold_size, used_folds, current_fold_path, exhausted_paths):
        fold_gen = self.comb_generator(available_groups, fold_size)
        while True:
            try:
                fold = tuple(sorted(next(fold_gen)))
                potential_path = tuple(current_fold_path + [fold])
                if fold not in used_folds and potential_path not in exhausted_paths:
                    return fold
            except StopIteration:
                return None
                
    def find_next_folds_(self, unique_groups, groups_per_fold, used_folds):
        current_fold_path = [] # folds that are currently being considered form a path, e.g. fold 1 -> fold 2 -> fold 3
        available_groups = set(unique_groups)
        exhausted_paths = set()
        i = 0
        while len(current_fold_path) < self.n_splits:
            fold_ix = len(current_fold_path)
            fold_size = groups_per_fold[fold_ix]
            # exhausted in current branch
            next_fold = self.find_next_fold_(available_groups, fold_size, used_folds, current_fold_path, exhausted_paths)

            if next_fold is None:
                if len(current_fold_path) == 0:
                    raise ValueError("The 'groups' parameter contains too few unique groups to create folds.")

                exhausted_paths.add(tuple(current_fold_path))
                exhausted_fold = current_fold_path.pop()
                available_groups.update(exhausted_fold)

                i += 1 # only update i if we backtrack
                if i > self.max_iter:
                    raise ValueError("Max iterations reached.")

            else:
                current_fold_path.append(next_fold)
                available_groups -= set(next_fold)
        
        used_folds.update(current_fold_path)
        return current_fold_path

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        if n_groups < self.n_splits:
            raise ValueError("Number of groups must be at least equal to n_splits.")
        
        # Calculate the number of groups per fold
        groups_per_fold = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        groups_per_fold[:n_groups % self.n_splits] += 1

        # Check if there are enough unique combinations for the smallest group
        smallest_fold_size = np.min(groups_per_fold)
        n_smallest_folds = np.sum(groups_per_fold == smallest_fold_size)
        smallest_fold_combinations = comb(n_groups, smallest_fold_size, exact=True)
        
        # Ensure there are enough combinations to form unique folds for each repeat
        if smallest_fold_combinations < self.n_repeats * n_smallest_folds:
            raise ValueError(f"Not enough unique folds for the requested number of repeats. Combnations (for the smallest fold size)={smallest_fold_combinations} < {self.n_repeats*n_smallest_folds=}.")
        
        used_folds = set()
        for ri in range(self.n_repeats):
            folds = self.find_next_folds_(unique_groups, groups_per_fold, used_folds)
            for fold in folds:
                test_idx = np.isin(groups, fold)
                train_idx = ~test_idx
                yield np.where(train_idx)[0], np.where(test_idx)[0]