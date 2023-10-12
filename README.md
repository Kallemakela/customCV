# Custom CV

Currently includes:

```python
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
    Repeated group CV iterator with no overlap.
    """
```

# Usage

# Configuration

# Documentation

