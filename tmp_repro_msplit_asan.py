import os
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from split import MSPLIT

X, y = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto', return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = MSPLIT(
    lookahead_depth_budget=3,
    full_depth_budget=4,
    reg=0.0,
    branch_penalty=0.0,
    max_bins=255,
    min_samples_leaf=8,
    min_child_size=8,
    proposal_atom_cap=32,
    max_branching=3,
    time_limit=10800,
    verbose=False,
    random_state=0,
    use_cpp_solver=True,
    interval_partition_solver='rush_dp',
    approx_mode=True,
    approx_ref_shortlist_enabled=True,
    approx_ref_widen_max=1,
    approx_challenger_sweep_enabled=False,
    approx_distilled_mode=True,
    approx_distilled_geometry_mode='teacher_atomcolor',
)

t0 = time.time()
clf.fit(X_train, y_train)
print('fit_sec', time.time()-t0)
print('done')
