# Custom CV

Additional cross-validation iterators for scikit-learn. Useful for group CV that require a lot of folds and repetitions due to high variance in validation and test scores (usually due to small dataset size).

Currently includes:

```python
class RepeatedUniqueFoldGroupKFold:
    """
    Repeated GroupKFold with unique folds across all repeats for even or uneven fold sizes.

    Each group appears in the test set exactly once per repeat.
    All folds are unique, even across repeats.
    Randomizaton works by generating a mapping from the original groups to a random permutation of the groups on each repeat.
    
    The fold search is performed in a depth-first manner, by first finding a unique fold from groups that have not yet been used on this repeat, adding it to a list of potential folds, and moving on to the next fold.
    If the fold search ends up in a situation where it is not possible to generate valid folds from the remaining groups, it backtracks by one fold and marks the backtracked fold as exhausted.
    """

class GroupShuffleSplit_(GroupShuffleSplit):
    """GroupShuffleSplit that accepts the shuffle parameter in the constructor so that it can be used with _RepeatedSplits"""

class RepeatedGroupShuffleSplit(_RepeatedSplits):
    """
    Randomized group CV iterator with possible overlap. 
    """

class RandomGroupKfold(GroupKFold):
    """GroupKFold that accepts the shuffle parameter. Ignores group sizes."""

class RepeatedGroupKfold(_RepeatedSplits):
    """
    Repeated group CV iterator with no overlap (per repeat).
    """
```

# Usage

# Configuration

# Documentation

