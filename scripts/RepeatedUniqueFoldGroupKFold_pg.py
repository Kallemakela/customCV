#%%
#!%load_ext autoreload
#!%autoreload 2
#%%
import numpy as np
from collections import defaultdict
import numpy as np
from scipy.special import comb
from itertools import combinations


import numpy as np
from itertools import combinations, chain
from customCV.group import RepeatedUniqueFoldGroupKFold, RepeatedUniqueFoldGroupKFoldPG

# #%%
xsize = 348
n_subjects = 8
k = n_subjects // 2
n_repeats = 7

cross_cv = RepeatedUniqueFoldGroupKFold(
    n_splits=k,
    n_repeats=n_repeats,
    random_state=0,
    max_iter=100
)

Xc = np.arange(xsize)
yc = np.arange(xsize)
subject_ids = np.random.randint(0, n_subjects, size=xsize)

train_count = defaultdict(int)
test_count = defaultdict(int)
for split_ix, (train_ix, test_ix) in enumerate(cross_cv.split(Xc, yc, subject_ids)):
    repeat_ix = split_ix // k
    train_subjects = np.unique(subject_ids[train_ix])
    test_subjects = np.unique(subject_ids[test_ix])
    # print(f'{split_ix:5d} {len(train_subjects)=}, {len(test_subjects)=}')
    print(f'{list(test_subjects)}')#, {list(train_subjects)}')
    for subject in train_subjects:
        train_count[subject] += 1
    for subject in test_subjects:
        test_count[subject] += 1

    if split_ix % k == k-1:
        print()

for s in np.unique(subject_ids):
    trs = train_count[s]
    ts = test_count[s]
    print(f'{s} {trs=} {ts=}')
    if ts != n_repeats and trs != n_repeats * (k-1):
        print(f"{s} should be in test {n_repeats=} times and in train {n_repeats*(k-1)=} times")
        print(f'{s} {trs=} {ts=}')
# %%
groups = np.arange(12) // 2
X = np.ones(len(groups))
y = np.ones(len(groups))
seeds = [
    # 3,
    11,
]
from itertools import combinations
for seed in seeds:
    cv = RepeatedUniqueFoldGroupKFold(n_splits=3, n_repeats=5, random_state=seed)
    print(f'{seed=}')
    all_possible_folds = list(combinations(range(len(set(groups))), 2))
    cv.all_possible_folds = all_possible_folds
    splits = list(cv.split(X, y, groups))
#%%