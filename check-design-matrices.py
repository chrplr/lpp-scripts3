#! /usr/bin/env python
# Time-stamp: <2018-04-08 14:01:20 cp983411>

""" report various stats for design matrix """
import sys
import numpy as np
import numpy.linalg
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")


for f in sys.argv[1:]:  # filenames are read on the command line
    print('\n\n# ' + f)
    dtxmat = pd.read_csv(f)
    dtxmat['constant'] = pd.Series(np.ones(dtxmat.shape[0]))
    m = dtxmat.as_matrix()
    print('\n## Means and SD')
    for c in dtxmat:
        print("%-15s" % c, round(dtxmat[c].mean(), 3), round(dtxmat[c].std(), 3))
    print("\n## Pairwise Correlations")
    corr = dtxmat.corr()
    print(round(corr, 3))
    print("\n## Condition Number", round(numpy.linalg.cond(m, p=None),2))
    print("\n## Variance inflation factors:")
    vifs = [vif(m, i) for i in range(m.shape[1])]
    for i, label in enumerate(dtxmat.columns):
        print("%-15s" % label, round(vifs[i], 3))
