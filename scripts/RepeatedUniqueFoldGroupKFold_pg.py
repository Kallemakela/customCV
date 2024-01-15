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
from customCV.group import RepeatedUniqueFoldGroupKFold

# #%%
xsize = 10000
n_subjects = 31
k = n_subjects // 2
n_repeats = 2

cross_cv = RepeatedUniqueFoldGroupKFold(
    n_splits=k,
    n_repeats=n_repeats,
    random_state=1,
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
    # print(f'{s} {trs=} {ts=}')
    if ts != n_repeats and trs != n_repeats * (k-1):
        print(f'{s} {trs=} {ts=}')
# %%
