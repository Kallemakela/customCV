import numpy as np


def get_allocation(y, n_splits):
    """
    Determine the optimal number of samples from each class in each fold,
    using round robin over the sorted y. (This can be done direct from
    counts, but that code is unreadable.)
    Initial approximation of how many members of each class
    """
    y_order = np.sort(y)
    n_classes = len(np.unique(y))
    allocation = np.asarray(
        [
            np.bincount(y_order[i::n_splits], minlength=n_classes)
            for i in range(n_splits)
        ]
    )
    return allocation
