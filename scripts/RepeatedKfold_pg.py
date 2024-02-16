#%%
import numpy as np
from collections import defaultdict
from customCV.repeated import RepeatedUniqueFoldKFold
from sklearn.model_selection import RepeatedKFold

k = 2
n_repeats = 3
cv = RepeatedUniqueFoldKFold(
    n_splits=k,
    n_repeats=n_repeats,
    random_state=1,
)
xsize = 4
Xc = np.arange(xsize)
yc = np.arange(xsize)
splits = list(cv.split(Xc, yc))
for split_ix, (train_ix, test_ix) in enumerate(splits):
    print(f'{split_ix:5d} {test_ix=}')
# %%
