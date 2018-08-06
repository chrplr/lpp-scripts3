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




def get_design_matrices(rootdir, model):
    matrices = []
    for j in range(1, 10):
        dmtx = pd.read_csv(os.path.join(rootdir, 'lpp-scripts3/outputs/design-matrices/en/{}/dmtx{}.csv'.format(model, j)))
        const = np.ones((dmtx.shape[0], 1))
        data = np.hstack((dmtx, const))
        matrices.append(data)
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
        logcsvwriter.writerow([loglabel, alpha, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, alpha, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])
    
    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None

    logo = LeaveOneGroupOut()
    outer_cv = logo.split(fmri_runs, groups=range(1,10))
    for train_outer, test_outer in outer_cv:
        fmri_train_outer = np.vstack(fmri_runs[r] for r in train_outer)
        predictors_train_outer = np.vstack([matrices[m] for m in train_outer])
        predictors_test_outer = np.vstack([matrices[m] for m in test_outer])
        fmri_test_outer = np.vstack(fmri_runs[r] for r in test_outer)
        inner_cv = logo.split(train_outer, groups=range(1,9))
        for train_inner, test_inner in inner_cv:
            fmri_train_inner = np.vstack(fmri_train_outer[r] for r in train_inner)
            predictors_train_inner = np.vstack(predictors_train_outer[m] for m in train_inner)
            # reg = RidgeCV(alphas=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0])
            reg = RidgeCV(alphas=[1.0, 10.0])
            reg.fit(predictors_train_inner, fmri_train_inner)
            global alpha
            alpha = reg.alpha_
            model = Ridge(alpha=alpha).fit(predictors_train_outer, fmri_train_outer)
            rsquares_training = clean_rscores(r2_score(fmri_train_outer,
                                                   model.predict(predictors_train_outer),    multioutput='raw_values'),
                                          .0, .99)
            r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, rsquares_training])

       # rsquares_test = clean_rscores(r2_score(test_outer,
        #                                       model.predict(design_matrices[test_outer]), multioutput='raw_values'),
             #                         .0, .99)

        rsquares_test = clean_rscores(r2_score(fmri_test_outer,
                                               model.predict(predictors_test_outer), multioutput='raw_values'),
                                      .0, .99)
            
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, rsquares_test])
        
    return (np.mean(r2_train, r2_test, axis=0))

    

if __name__ == '__main__':
 
    rootdir = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI"
    #rootdir = "/media/sneza/FREECOM HDD/LePetitPrince_Pallier_2018/MRI"
    model = ' '.join(sys.argv[1:])
    matrices = get_design_matrices(rootdir, model)
    #alpha = 0.4
    #alpha = alpha
    #subjects = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
    #subjects = [71]
    subjects = [57]

    for subject in subjects:
    
        masker = compute_global_masker(rootdir, [subject])
    
        fmri_filenames = sorted(glob.glob(os.path.join(rootdir, 
                                                   "fmri-data/en",
                                                   "sub-%03d" % subject, 
                                                   "func", 
                                                   "resample*.nii")))
        fmri_runs = [masker.transform(f) for f in fmri_filenames]
    
        loglabel = subject
        logcsvwriter = csv.writer(open(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "try_nested_stats_{}_subject_{}.log".format(model, subject)), "a+"))
        
        

        # Compute and save training and test of the full model
        r2train, r2test = Parallel(n_jobs=-2)(delayed(compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)))
        nib.save(masker.inverse_transform(r2train), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_{}_alpha_{}_train_sub_{:03}.nii".format(model, alpha, subject)))

        nib.save(masker.inverse_transform(r2test), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_{}_alpha_{}_test_sub_{:03}.nii".format(model, alpha, subject)))
 
 
        ### Compare the full model with the model where embeddings are dropped
        
        # Drop embeddings 
        without_embeddings = [np.delete(mtx, np.s_[:300], 1) for mtx in matrices]
        r2train_simple, r2test_simple = compute_crossvalidated_r2(fmri_runs, without_embeddings,  loglabel, logcsvwriter)
        
        # Save the simple model (without embeddings)
        nib.save(masker.inverse_transform(r2test_simple), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_{}_alpha_{}_rms_wordrate_freq_bottomup_sub_{:03}.nii".format(model, alpha, subject)))
        
        # Compute the difference and save 
        r2train_embeddings_only = r2train - r2train_simple
        nib.save(masker.inverse_transform(r2train_embeddings_only), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_{}_alpha_{}_embeddings_only_train_sub_{:03}.nii".format(model, alpha, subject)))
        r2test_embeddings_only = r2test - r2test_simple
        
        
        # Plot glass brain of the final result (embeddings test)
        img = mean_img(masker.inverse_transform(r2test_embeddings_only))
        display = None
        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, title='Variance explained by word embeddings')
        display.savefig(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_{}_alpha_{}_embeddings_only_test_sub_{:03}.png".format(model, alpha, subject)))
        display.close()
        # Save embeddings test
        nib.save(masker.inverse_transform(r2test_embeddings_only), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_{}_alpha_{}_embeddings_only_test_sub_{:03}.nii".format(model, alpha, subject)))
