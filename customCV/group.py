import numpy as np
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

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.random_group_map = None

    def find_next_fold_(self, available_groups, fold_size, used_folds, exhausted=[]):
            fold_gen = combinations(available_groups, fold_size)
            while True:
                try:
                    fold = tuple(sorted(next(fold_gen)))
                    if self.random_group_map is not None:
                        fold = tuple(sorted(self.random_group_map[g] for g in fold))
                    if fold not in used_folds and fold not in exhausted:
                        return fold
                except StopIteration:
                    return None
                
    def find_next_folds_(self, unique_groups, groups_per_fold, used_folds):
            potential_folds = []
            available_groups = unique_groups.copy()

            while len(potential_folds) < self.n_splits:
                exhausted = []
                fold_ix = len(potential_folds)
                fold_size = groups_per_fold[fold_ix]
                next_fold = self.find_next_fold_(available_groups, fold_size, used_folds, exhausted)
                if next_fold is None:
                    if len(potential_folds) == 0:
                        raise ValueError("The 'groups' parameter contains too few unique groups to create folds.")
                    exhausted_fold = potential_folds.pop()
                    exhausted.append(exhausted_fold)
                    
                    if self.random_group_map is not None:
                        exhausted = tuple(sorted(self.random_group_map_inv[g] for g in exhausted_fold))

                    available_groups = np.concatenate([available_groups, exhausted_fold])
                else:
                    potential_folds.append(next_fold)
                    
                    if self.random_group_map is not None:
                        next_fold = tuple(sorted(self.random_group_map_inv[g] for g in next_fold))

                    available_groups = np.setdiff1d(available_groups, next_fold)
            
            used_folds.update(potential_folds)
            return potential_folds

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
        min_group_count = np.min(groups_per_fold)
        n_min_groups = np.sum(groups_per_fold == min_group_count)
        min_group_combinations = comb(n_groups, min_group_count, exact=True)
        
        # Ensure there are enough combinations to form unique folds for each repeat
        if min_group_combinations < self.n_repeats * n_min_groups:
            print(f"{min_group_combinations=} >= {self.n_repeats*n_min_groups=}")
            raise ValueError("Not enough unique folds for the requested number of repeats.")
        
        used_folds = set()
        for ri in range(self.n_repeats):

            if self.random_state is not None:
                rng = np.random.default_rng(self.random_state + ri)
                self.random_group_map = dict(zip(unique_groups, rng.permutation(unique_groups)))
                self.random_group_map_inv = {v: k for k, v in self.random_group_map.items()}

            folds = self.find_next_folds_(unique_groups, groups_per_fold, used_folds)
            for fold in folds:
                test_idx = np.isin(groups, fold)
                train_idx = ~test_idx
                yield np.where(train_idx)[0], np.where(test_idx)[0]