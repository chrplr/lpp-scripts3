#! /usr/bin/env python3
# Time-stamp: <2019-05-16 13:35:36 christophe@pallier.org>

import pandas as pd
import nibabel as nib
from nilearn.image import math_img, mean_img, threshold_img
from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold
from nilearn.plotting import plot_glass_brain


def one_sample_ttest(filenames, name):
    design_matrix = pd.DataFrame([1] * len(filenames), columns=['intercept'])
    second_level_model = SecondLevelModel().fit(filenames, design_matrix=design_matrix)
    z_map = second_level_model.compute_contrast(output_type='z_score')
    nib.save(zmap, name + '.nii')
    thmap, threshold1 = map_threshold(z_map, level=.001, height_control='fpr', cluster_threshold=10)
    display = plot_glass_brain(thmap, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(name + '.png')
    display.close()


def compute_diffmaps(scans_A, scans_B):
    assert len(scans_A) == len(scans_B)
    list_fnames = []
    for sub, a, b in enumerate(zip(scans_A, scans_B)):
        diff =  math_img('a - b', img1=a, img=b)
        fname = f'diff_{sub:03}'
        nib.save(fname, diff)
        list_fnames.append(fname)
    return fnames


def compare_2conds_within_subjects(scans_A, scans_B, name):
    files = compute_diffmaps(scans_A, scans_B)
    one_sample_ttest(files, name)



