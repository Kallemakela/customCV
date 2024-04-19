import numpy as np
from collections import defaultdict
import random
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from scipy.special import comb
from itertools import combinations
from customCV.repeated import RepeatedUniqueFoldKFold, RepeatedUniqueFoldKFoldPG

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

class GroupCVWrapper:
    def __init__(self, base_cv):
        """
        Wraps a non-group cross-validator to make it group-aware.

        Parameters:
        - base_cv: The non-group KFold cross-validator instance from scikit-learn.
        - random_state: Random state for reproducibility, if applicable to the base_cv.
        """
        self.base_cv = base_cv

    def _groups_to_indices(self, groups):
        """Map each group to the indices of samples belonging to that group."""
        group_indices = defaultdict(list)
        for index, group in enumerate(groups):
            group_indices[group].append(index)
        return group_indices

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets, ensuring that
        samples from the same group are not split across folds.

        Parameters:
        - X: Features, placeholder for compatibility, not used.
        - y: Targets, placeholder for compatibility, may be used by the base_cv if it requires.
        - groups: Array-like, with the same length as `X`, indicating group membership for each sample.

        Yields:
        - train, test: Indices for training and test sets, respectively.
        """
        if groups is None:
            raise ValueError("Groups must be specified for GroupKFoldWrapper")

        # Map groups to indices
        group_indices = self._groups_to_indices(groups)

        # Flatten group indices to create a pseudo-X for generating base_cv splits,
        # where each "sample" in this pseudo-X represents a group.
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        pseudo_X = np.arange(len(unique_groups)).reshape(-1, 1)

        # Generate splits using the base_cv on the pseudo-X,
        # then map back to original indices.
        for train_groups_idx, test_groups_idx in self.base_cv.split(pseudo_X):
            train_idx = np.concatenate([group_indices[unique_groups[i]] for i in train_groups_idx])
            test_idx = np.concatenate([group_indices[unique_groups[i]] for i in test_groups_idx])
            yield train_idx, test_idx


class RepeatedUniqueFoldGroupKFold:
    """Group extension of RepeatedUniqueFoldKFold"""
    def __init__(self, n_splits=5, n_repeats=10, random_state=42, max_iter=int(1e6), **kwargs):
        self.n_splits, self.n_repeats, self.random_state, self.max_iter = n_splits, n_repeats, random_state, max_iter
        self.base_cv = RepeatedUniqueFoldKFold(n_splits, n_repeats, random_state, max_iter, **kwargs)
        self.wrapper = GroupCVWrapper(self.base_cv)

    def split(self, X, y=None, groups=None):
        return self.wrapper.split(X, y, groups)
        
class RepeatedUniqueFoldGroupKFoldPG:
    """Group extension of RepeatedUniqueFoldKFoldPG"""
    def __init__(self, n_splits=5, n_repeats=10, random_state=42, max_iter=int(1e6), **kwargs):
        self.n_splits, self.n_repeats, self.random_state, self.max_iter = n_splits, n_repeats, random_state, max_iter
        self.base_cv = RepeatedUniqueFoldKFoldPG(n_splits, n_repeats, random_state, max_iter, **kwargs)
        self.wrapper = GroupCVWrapper(self.base_cv)

    def split(self, X, y=None, groups=None):
        return self.wrapper.split(X, y, groups)

