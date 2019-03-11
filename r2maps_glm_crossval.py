#! /usr/bin/env python3
# Time-stamp: <>

import glob
import os
import os.path as op
import sys
import csv
import pandas as pd
import seaborn as sns

import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker

from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from nilearn.image import math_img, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import coord_transform
import pylab as plt
from joblib import Parallel, delayed


def get_fmri_files_from_subject(rootdir, subject):
    return sorted(glob.glob(os.path.join(rootdir,
                                         "fmri-data/en",
                                         "sub-%03d" % subject,
                                         "func",
                                         "resamp-4.0*.nii")))


def get_design_matrices(modeldir):
    matrices = []
    for j in range(1, 10):
        path = op.join(modeldir, f'dmtx_{j}.csv')
        dmtx = pd.read_csv(path, header=0)
        # add the constant)
        const = np.ones((dmtx.shape[0], 1))
        dmtx = np.hstack((dmtx, const))
        matrices.append(dmtx)
    return matrices


def compute_global_masker(rootdir, subjects):
    files = [glob.glob(os.path.join(rootdir,
                                    "fmri-data/en",
                                    "sub-%03d" % s,
                                    "func/resamp-4.0*.nii")) for s in subjects]
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks))
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
    masker.fit()
    return masker


def clean_rscores(rscores, r2min, r2max): # remove values with are too low and values too good to be true (e.g. voxels without variation)
    return np.array([0 if (x < r2min or x >= r2max) else x for x in rscores])


def compute_crossvalidated_r2(fmri_runs, design_matrices, loglabel, logcsvwriter):

    def log(r2_train, r2_test):
        """ log stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, 'training', np.mean(r2_train), np.std(r2_train),
                               np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, 'test', np.mean(r2_test), np.std(r2_test),
                               np.min(r2_test), np.max(r2_test)])

    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None

    logo = LeaveOneGroupOut()#leave on run out !
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data_train = np.vstack([fmri_runs[i] for i in train])
        predictors_train = np.vstack([design_matrices[i] for i in train])
        model = LinearRegression().fit(predictors_train, fmri_data_train)

        rsquares_training = r2_score(fmri_data_train,
                                     model.predict(predictors_train),
                                     multioutput='raw_values')
        rsquares_training = clean_rscores(rsquares_training, .0, .99)

        test_run = test[0]
        rsquares_test = r2_score(fmri_runs[test_run],
                                 model.predict(design_matrices[test_run]),
                                 multioutput='raw_values')
        rsquares_test = clean_rscores(rsquares_test, .0, .99)

        log(rsquares_training, rsquares_test)

        r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, rsquares_training])
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, rsquares_test])

    return (np.mean(r2_train, axis=0), np.mean(r2_test, axis=0))


def do_single_subject(rootdir, subj, matrices):
    fmri_filenames = get_fmri_files_from_subject(rootdir, subj)
    fmri_runs = [masker.transform(f) for f in fmri_filenames]

    loglabel = subj
    logcsvwriter = csv.writer(open("test.log", "a+"))

    r2train, r2test = compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)

    r2train_img = masker.inverse_transform(r2train)
    nib.save(r2train_img, f'train_{subj:03d}.nii.gz')

    r2test_img = masker.inverse_transform(r2test)
    nib.save(r2test_img, f'test_{subj:03d}.nii.gz')

    # compute the increase in R2 due to each predictor
    # for reg in range(MATRICES[0].shape[1]):
    #     """ remove one predictor from the design matrix in test to compare it with the full model """
    #     new_design_matrices = [np.delete(mtx, reg, 1) for mtx in MATRICES]
    #     r2train_dropped, r2test_dropped = compute_crossvalidated_r2(fmri_runs, new_design_matrices, loglabel, logcsvwriter)

    #     nib.save(masker.inverse_transform(r2train_dropped),
    #              'train_dropping_{}_{:03}.nii'.format(reg,subject))

    #     nib.save(masker.inverse_transform(r2test_dropped),
    #              'test_dropping_{}_{:03}.nii'.format(reg,subject))

    #     r2train_difference = r2train - r2train_dropped
    #     nib.save(masker.inverse_transform(r2train_difference),
    #              'train_r2_increase_when_adding_{}_{:03}.nii'.format(reg,subject))

    #     r2test_difference = r2test - r2test_dropped
    #     nib.save(masker.inverse_transform(r2test_difference),
    #              'test_r2_increase_when_adding_{}_{:03}.nii'.format(reg,subject))


    display = plot_glass_brain(r2test_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(f'test_{subj:03}.png')
    display.close()

    display = plot_glass_brain(r2train_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(f'train_{subj:03}.png')
    display.close()


if __name__ == '__main__':

    DEBUG = True
    PARALLEL = True

    ROOTDIR = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI"
    MODELDIR = 'lpp-scripts3/outputs/design-matrices/en/'
    MODEL = 'rms-wordrate-freq-topdown'
    OUTDIR = op.join(ROOTDIR, 'outputs/r2maps-glm', MODEL)

    olddir = os.getcwd()
    os.chdir(OUTDIR)

    MATRICES = get_design_matrices(op.join(ROOTDIR, MODELDIR, MODEL))

    subjects = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75,
                76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93,
                94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109,
                110, 113, 114, 115]

    if DEBUG:
        subjects = subjects[:5]  # test on first subjects only

    print(f'Computing global mask...')
    masker = compute_global_masker(ROOTDIR, subjects)  # sloow


    if PARALLEL:
        Parallel(n_jobs=-1)(delayed(do_single_subject)(ROOTDIR, sub, MATRICES) for sub in subjects)
    else:
        for sub in subjects:
            print(f'Processing subject {sub:03d}...')
            do_single_subject(ROOTDIR, sub, MATRICES)


    os.chdir(olddir)
