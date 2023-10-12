import numpy as np
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

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
