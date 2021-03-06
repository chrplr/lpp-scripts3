# IMPORTS
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker

import glob
import os
import os.path
import sys
import nibabel as nib

import csv
import pandas as pd
import seaborn as sns

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut

from nilearn.image import math_img, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import coord_transform
import pylab as plt
from joblib import Parallel, delayed




def get_design_matrices(rootdir):
    matrices = []
    for j in range(1, 10):
        data = pd.read_csv(os.path.join(rootdir, 'scripts-python/design-matrices/en/rms-wordrate-freq-bottomup/dmtx_{}.csv'.format(j)), header=None)
        dmtx = data[1:]
        const = np.ones((dmtx.shape[0], 1))
        dmtx = np.hstack((dmtx, const))
        matrices.append(dmtx)
    return matrices

def compute_global_masker(rootdir, subjects):
    masks = [compute_epi_mask(glob.glob(os.path.join(rootdir, "fmri-data/en", "sub-%03d" % s, "func","*.nii"))) for s in subjects]
    global_mask = math_img('img>0.5', img=mean_img(masks))
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
    masker.fit()
    return masker

def clean_rscores(rscores, r2min, r2max): # remove negative values (worse than the constant model) and values too good to be true (e.g. voxels without variation)
    return np.array([0 if (x < r2min or x >= r2max) else x for x in rscores])
   
def compute_crossvalidated_r2(fmri_runs, design_matrices, loglabel, logcsvwriter):
    
    def log(r2_train, r2_test):
        """ just logging stats per fold to a csv file """
#        logcsvwriter.writerow([loglabel, alpha, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, alpha, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])
    
    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None

    # estimate alpha by gridsearch
    predictors = np.vstack(d for d in design_matrices)
    predictors_test = predictors
    predictors_test[:, 1:] = 0 # set all but the first column (rms) to zero
    predictors_test = predictors_test.reshape(1, -1) #Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.


    data = np.vstack(r for r in fmri_runs)
    #reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    reg = RidgeCV(alphas=[0.5, 1.0, 3.0, 5.0])
    reg.fit(predictors, data)
    alpha = reg.alpha_
    print('alpha: ',alpha)
    
    
    logo = LeaveOneGroupOut()
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data = np.vstack([fmri_runs[i] for i in train])
        print('fmri_data: ',fmri_data)
        predictors = np.vstack([design_matrices[i] for i in train])
        print('predictors: ',predictors)
        model = Ridge(alpha=alpha).fit(predictors, fmri_data)
            
#        rsquares_training = clean_rscores(r2_score(fmri_data, 
#                                                   model.predict(predictors_test), multioutput='raw_values'), 
#                                          .0, .99)
        test_run = test[0]
        print('test_run: ',test_run)
        print('predictors_test[test_run]: ',predictors_test[test_run])
        rsquares_test = clean_rscores(r2_score(fmri_runs[test_run], 
                                               model.predict(predictors_test[test_run]), multioutput='raw_values'),
                                      .0, .99)
        
        log(rsquares_training, rsquares_test)

#        r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, rsquares_training])    
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, rsquares_test])
        
    return (np.mean(r2_test, axis=0))
    
    

if __name__ == '__main__':
 
    rootdir = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI"
    matrices = get_design_matrices(rootdir)
    
    subjects = [57]
    # english subjects: 57 58 59 61 62 63 64 65 66 67 68 69 71 72 73 74 75 76 77 78 79 80 81 82 83 84 86 87 88 89 91 92 93 94 95 96 97 98 99 100 101 103 104 105 106 108 109 110 113 114 115
    for subject in subjects:
   #   alphas = [0.001]
    #alpha = alphas[0]
    
        masker = compute_global_masker(rootdir, [subject])
    
        fmri_filenames = sorted(glob.glob(os.path.join(rootdir, 
                                                   "fmri-data/en",
                                                   "sub-%03d" % subject, 
                                                   "func", 
                                                   "resample*.nii")))
        fmri_runs = [masker.transform(f) for f in fmri_filenames]
    
        loglabel = subject
        logcsvwriter = csv.writer(open(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/42B_en", "second_method.log"), "a+"))

    
        r2scores = compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)
#        nib.save(masker.inverse_transform(r2train), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/42B_en", "second_method_train_sub_{:03}.nii".format(subject)))

        nib.save(masker.inverse_transform(r2test), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/42B_en", "second_method_test_sub_{:03}.nii".format(subject)))
        img = mean_img(masker.inverse_transform(r2test))
        display = None
        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, title='rms when only in test')
        display.savefig(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/42B_en", "rms_second_sub_{:03}.png".format(subject)))
        display.close()

 
