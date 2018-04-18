#! /usr/bin/env python
# Time-stamp: <2018-04-18 14:56:19 cp983411>

import os
import sys
import os.path as op
import glob
import getopt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import nibabel
from nistats.second_level_model import SecondLevelModel
from nilearn import plotting
from scipy.stats import norm
import matplotlib.pyplot as plt


def create_one_sample_t_test(name, maps, output_dir, smoothing_fwhm=8.0):
    if not op.isdir(output_dir):
        op.mkdir(output_dir)

    model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
    design_matrix = pd.DataFrame([1] * len(maps),
                                 columns=['intercept'])
    model = model.fit(maps,
                      design_matrix=design_matrix)
    z_map = model.compute_contrast(output_type='z_score')
    nibabel.save(z_map, op.join(output_dir, "{}_group_zmap.nii.gz".format(name)))

    p_val = 0.001
    z_th = norm.isf(p_val)
    z_th = 5.5
    display = plotting.plot_glass_brain(
        z_map, threshold=z_th,
        colorbar=True,
        plot_abs=False,
        display_mode='lzry',
        title=name)
    display.savefig(op.join(output_dir, "{}_group_zmap".format(name)))

if __name__ == '__main__':
    # defaults
    data_dir = os.getenv('DATA_DIR')
    output_dir = '.'
    
    # parse command line to change default
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:o:",
                                   ["data_dir=", "output_dir="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o in ('-d', '--data_dir'):
            data_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a

    assert(data_dir is not None)
    
    cons = ('topdownO', 'bottomupO', 'f0O', 'wordrateO', 'freqO', 'rms')
#    cons = os.getenv('REGS').split()
    for con in cons:
        mask = op.join(data_dir, '%s_*effsize.nii.gz' % con)
        maps = glob.glob(mask)
        if maps == []:
            print("Warning: %s : no such files" % mask)
        else:
            create_one_sample_t_test(con, maps, output_dir)
