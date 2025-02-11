import numpy as np
import pytest
from customCV.group import GroupCVWrapper
from sklearn.model_selection import KFold


def test_kfold():
    groups = np.array([0, 0, 1, 0])
    kfold = KFold(n_splits=2)
    cv = GroupCVWrapper(kfold)
    for train, test in cv.split(groups=groups):
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        assert len(test_groups) == 1
        assert len(train_groups) == 1
        assert test_groups.isdisjoint(train_groups)
