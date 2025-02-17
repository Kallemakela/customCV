import numpy as np
from sklearn.model_selection import StratifiedKFold


class RSUKfold:
    """
    Brute force implementation of repeated stratified unique k-fold cross-validation.

    Basically runs scikit-learn's StratifiedKFold in a loop until it finds the desired number of unique folds.
    """

    def __init__(self, n_splits=5, n_repeats=2, random_state=42, max_tries=int(1e4)):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.max_tries = max_tries

        if random_state is None:
            raise ValueError("random_state must be set to an integer value.")

    def get_repeat(self, X, y, ti):
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state + ti,
        )
        iter_folds = []
        for _, test_index in skf.split(X, y):
            test_fold = tuple(sorted(test_index))
            if test_fold in self.seen_folds:
                self.failed_tries += 1
                return None
            iter_folds.append(test_fold)
        return iter_folds

    def split(self, X, y):
        self.folds = []
        self.seen_folds = set()
        self.failed_tries = 0
        for ti in range(self.max_tries):
            repeat_folds = self.get_repeat(X, y, ti)
            if repeat_folds is not None:
                self.folds.extend(repeat_folds)
                self.seen_folds.update(repeat_folds)
            if len(self.folds) == self.n_splits * self.n_repeats:
                break

        print(len(self.folds), self.n_splits * self.n_repeats)
        if len(self.folds) != self.n_splits * self.n_repeats:
            raise ValueError(
                f"Failed to generate {self.n_splits * self.n_repeats} unique folds after {self.max_tries} tries."
            )

        # Yield the folds
        for ri in range(self.n_repeats):
            for fi in range(self.n_splits):
                test_idx = np.array(self.folds[ri * self.n_splits + fi])
                train_idx = np.array(
                    [
                        i
                        for i in range(X.shape[0])
                        if i not in self.folds[ri * self.n_splits + fi]
                    ]
                )
                yield train_idx, test_idx
