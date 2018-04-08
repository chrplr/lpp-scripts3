#! /usr/bin/env python
# Time-stamp: <2018-04-06 16:32:24 cp983411>

"""
Extract data from contrasts maps in a set of ROIs
"""

import sys
from glob import glob
import getopt
import os
import os.path as op
import pandas as pd

from nilearn.input_data import NiftiMapsMasker
# from nilearn.plotting import plot_roi


def basenames(files):
    return [op.splitext(x)[0] for x in [op.basename(op.splitext(f)[0]) for f in files]]

if __name__ == '__main__':
    # defaults
    data_dir = os.getenv('DATA_DIR')
    images = '*effsize*.nii*'
    mask_dir = 'ROIs'
    output = 'rois.csv'
    
    # parse command line to change default
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:m:i:o:",
                                   ["data_dir=", "maskdir=", "images=", "output="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o in ('-d', '--data_dir'):
            data_dir = a
        elif o in ('-m', '--mask_dir'):
            mask_dir = a
        elif o in ('-i', '--images'):
            images = a
        elif o in ('-o', '--output'):
            output = a
                        

    filter = op.join(data_dir, images)
    images = sorted(glob(filter))
    if images == []:
        print('Empty list :' + filter)
        sys.exit(3)
        
    labels = basenames(images)
    u = [x.split('_') for x in labels]
    subj = [x[1] for x in u]
    con = [x[0] for x in u]
    
    ROIs = sorted(glob(op.join(mask_dir, '*.nii')))
    roi_names = basenames(ROIs)
    
    # extract data 
    masker = NiftiMapsMasker(ROIs)
    values = masker.fit_transform(images)

    # save it into a pandas DataFrame
    df = pd.DataFrame(columns=['subject', 'con', 'ROI', 'beta'])

    n1, n2 = values.shape
    k = 0
    for i1 in range(n1):
        for i2 in range(n2):
             df.loc[k] = pd.Series({'subject': subj[i1],
                                    'con': con[i1],
                                    'ROI': roi_names[i2],
                                    'beta': values[i1, i2]})
             k = k + 1
    df.to_csv(output, index=False)
