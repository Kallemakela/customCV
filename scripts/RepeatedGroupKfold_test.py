#%%
import numpy as np
from collections import defaultdict
from customCV.group import RepeatedGroupKfold
#%%
xsize = 10000
n_subjects = 4
k = 2
n_repeats = 2

cross_cv = RepeatedGroupKfold(
    n_splits=k,
    n_repeats=n_repeats,
    # random_state=1,
    # shuffle=True,
)

Xc = np.arange(xsize)
yc = np.arange(xsize)
subject_ids = np.random.randint(0, n_subjects, size=xsize)

train_count = defaultdict(int)
test_count = defaultdict(int)
for split_ix, (train_ix, test_ix) in enumerate(cross_cv.split(Xc, yc, subject_ids)):
    train_subjects = np.unique(subject_ids[train_ix])
    test_subjects = np.unique(subject_ids[test_ix])
    print(f'{split_ix:5d} {len(train_subjects)=}, {len(test_subjects)=}')
    print(f'{list(test_subjects)}')
    for subject in train_subjects:
        train_count[subject] += 1
    for subject in test_subjects:
        test_count[subject] += 1

for s in np.unique(subject_ids):
    trs = train_count[s]
    ts = test_count[s]
    # print(f'{s} {trs=} {ts=}')
    if ts != n_repeats and trs != n_repeats * (k-1):
        print(f'{s} {trs=} {ts=}')
# %%
