#! /usr/bin/env python
# Time-stamp: <2018-08-10 16:16:21 cp983411>

""" Compute crossvalided r2scores in each voxels, using a LinearModel """

from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker

import glob
import os
import os.path as op
import sys
import nibabel as nib

import csv
import pandas as pd
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression # package to do the r2 with glm

from nilearn.image import math_img, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import coord_transform
import pylab as plt
from joblib import Parallel, delayed


def get_design_matrices(rootdir, lang, model):
    matrices = []
    for j in range(1, 10):
        data = pd.read_csv(op.join(rootdir, 'lpp-scripts3/outputs/design-matrices/{}/{}/dmtx_{}.csv'.format(lang, model, j)), header=None)
        dmtx = data[1:]
        const = np.ones((dmtx.shape[0], 1))
        dmtx = np.hstack((dmtx, const))
        matrices.append(dmtx)
    return matrices


def compute_global_masker(rootdir, subjects):
    # masks = [compute_epi_mask(glob.glob(op.join(rootdir, "fmri-data/en", "sub-%03d" % s, "func","*.nii"))) for s in subjects]
    # global_mask = math_img('img>0.5', img=mean_img(masks))
    # masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True, memory='/volatile/tmp')
    masker = MultiNiftiMasker(mask_img=op.join(rootdir,
                                               "lpp-scripts3/inputs/ROIs/mask_ICV.nii"),
                              detrend=True,
                              standardize=True)
    masker.fit()
    return masker


def clean_rscores(rscores, r2min, r2max): # remove negative values (worse than the constant model) and values too good to be true (e.g. voxels without variation)
    return np.array([0 if (x < r2min or x >= r2max) else x for x in rscores])


def compute_crossvalidated_r2(fmri_runs, design_matrices, loglabel, logcsvwriter):
    
    def log(r2_train, r2_test):
        """ just logging stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])
    
    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None
    
    logo = LeaveOneGroupOut()
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data = np.vstack([fmri_runs[i] for i in train])
        predictors = np.vstack([design_matrices[i] for i in train])
        model = LinearRegression().fit(predictors, fmri_data)

        rsquares_training = clean_rscores(r2_score(fmri_data, 
                                                   model.predict(predictors), multioutput='raw_values'), 
                                          .0, .99)
        test_run = test[0]
        rsquares_test = clean_rscores(r2_score(fmri_runs[test_run], 
                                               model.predict(design_matrices[test_run]), multioutput='raw_values'),
                                      .0, .99)
        
        log(rsquares_training, rsquares_test)

        r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, rsquares_training])    
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, rsquares_test])
        
    return (np.mean(r2_train, axis=0), np.mean(r2_test, axis=0))
    

def do_subject(rootdir, subject, masker, outputdir):
    fmri_filenames = sorted(glob.glob(op.join(rootdir, 
                                              "fmri-data/en",
                                              "sub-%03d" % subject, 
                                              "func", 
                                              "resample*.nii")))
    
    fmri_runs = [masker.transform(f) for f in fmri_filenames]
    
    loglabel = subject
    logcsvwriter = csv.writer(open("test.log", "a+"))

    ### Compute and save training and test of the full model
    r2train, r2test = compute_crossvalidated_r2(fmri_runs,
                                                matrices,
                                                loglabel,
                                                logcsvwriter)
    
    nib.save(masker.inverse_transform(r2train), 
             op.join(outputdir, 'train_{:03}.nii'.format(subject)))
    nib.save(masker.inverse_transform(r2test), 
             op.join(outputdir, 'test_{:03}.nii'.format(subject))) 
 
    ### Compare the full model with the model where different regressors are dropped
        
    # remove each predictor sequentially from the design matrix in test to compare it with the full model
    for reg in range(matrices[0].shape[1]):
        # restricted model
        new_design_matrices = [np.delete(mtx, reg, 1) for mtx in matrices]
        r2train_dropped, r2test_dropped = compute_crossvalidated_r2(fmri_runs, new_design_matrices, loglabel, logcsvwriter)
        nib.save(masker.inverse_transform(r2train_dropped), 
                 op.join(outputdir, 'train_dropping_{}_{:03}.nii'.format(reg,subject)))
        nib.save(masker.inverse_transform(r2test_dropped), 
                 op.join(outputdir, 'test_dropping_{}_{:03}.nii'.format(reg,subject)))

        # Compute the difference between the full and restricted model
        r2train_difference = r2train - r2train_dropped
        nib.save(masker.inverse_transform(r2train_difference), 
                 op.join(outputdir, 'train_r2_increase_when_adding_{}_{:03}.nii'.format(reg,subject)))
        r2test_difference = r2test - r2test_dropped
        nib.save(masker.inverse_transform(r2test_difference), 
                 op.join(outputdir, 'test_r2_increase_when_adding_{}_{:03}.nii'.format(reg,subject)))
             
        ### Plot glass brain of the final result     
        img = masker.inverse_transform(r2test_difference)
        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, title='Variance explained by {}, subject{:03}'.format(reg, subject))
        display.savefig(op.join(outputdir, "test_r2_increase_when_adding_{:02}_sub_{:03}.png".format(reg, subject)))
        display.close()

if __name__ == '__main__':

    rootdir = os.getenv("ROOT_DIR")
    matrices = get_design_matrices(rootdir,
                                   os.getenv("LINGUA"),
                                   os.getenv("MODEL"))

    outputdir = op.join(rootdir,
                        "lpp-scripts3",
                        "outputs",
                        "r2maps-glm",
                        os.getenv("LINGUA"),
                        os.getenv("MODEL"))
                    
    subjects = [ 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
    
    # english subjects: 57 58 59 61 62 63 64 65 66 67 68 69 71 72 73 74 75 76 77 78 79 80 81 82 83 84 86 87 88 89 91 92 93 94 95 96 97 98 99 100 101 103 104 105 106 108 109 110 113 114 115

    masker = compute_global_masker(os.getenv("ROOT_DIR"), subjects)
    
    Parallel(n_jobs=-2)(delayed(do_subject)(rootdir, s, masker, outputdir) for s in subjects)
