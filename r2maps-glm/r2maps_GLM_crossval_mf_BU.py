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


#/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/design-matrices/en/rms-wordrate-freq-bottomup
#'scripts-python/design-matrices/en/rms-wordrate-freq-bottomup/dmtx_{}.csv'

def get_design_matrices(rootdir):
    matrices = []
    for j in range(1, 10):
        data = pd.read_csv(os.path.join(rootdir, 'lpp-scripts3/outputs/design-matrices/en/rms-wordrate-freq-bottomup/dmtx_{}.csv'.format(j)), header=None)
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
        logcsvwriter.writerow([loglabel, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])
    
    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None
    
    logo = LeaveOneGroupOut()#leave on run out !
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data = np.vstack([fmri_runs[i] for i in train])
        predictors = np.vstack([design_matrices[i] for i in train])
        model = LinearRegression().fit(predictors, fmri_data)
        #model = L(alpha=alpha).fit(predictors, fmri_data)
        #possible de modifier en faisant en GLM model = LinearModel().fit(predictors, fmri_data)
        # plus haut importer dans scikitlearn verifier l'ojbet de  LinearModel
            
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
    
    

if __name__ == '__main__':
 
    rootdir = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI"
    matrices = get_design_matrices(rootdir)
    
    #subjects = [57, 58,  59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, ]
    subjects = [ 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
    # 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115
    
    # english subjects: 57 58 59 61 62 63 64 65 66 67 68 69 71 72 73 74 75 76 77 78 79 80 81 82 83 84 86 87 88 89 91 92 93 94 95 96 97 98 99 100 101 103 104 105 106 108 109 110 113 114 115
    
    #subject = subjects[0]
    # alphas = [0.001]
    # alpha = alphas[0]
    #  subjects                    list                n=1
    
    for subject in subjects:
    
        masker = compute_global_masker(rootdir, [subject])
    
        fmri_filenames = sorted(glob.glob(os.path.join(rootdir, 
                                                   "fmri-data/en",
                                                   "sub-%03d" % subject, 
                                                   "func", 
                                                   "resample*.nii")))
        fmri_runs = [masker.transform(f) for f in fmri_filenames]
    
        loglabel = subject
        logcsvwriter = csv.writer(open("test.log", "a+"))
    #logcsvwriter = csv.writer(open(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "stats_{}_alpha_{}.log".format(model, alpha)), "a+"))

        ### Compute and save training and test of the full model
        r2train, r2test = compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)
    
        nib.save(masker.inverse_transform(r2train), 
             'train_{:03}.nii'.format(subject))

        nib.save(masker.inverse_transform(r2test), 
             'test_{:03}.nii'.format(subject))
 
 
        ### Compare the full model with the model where different regressors are dropped
        
        # Drop regressors 
        for reg in range(matrices[0].shape[1]):
            #""" remove one predictor from the design matrix in test to compare it with the full model """ 
            
            new_design_matrices = [np.delete(mtx, reg, 1) for mtx in matrices]
            r2train_dropped, r2test_dropped = compute_crossvalidated_r2(fmri_runs, new_design_matrices, loglabel, logcsvwriter)
            
            # Save the simple model (without a given regressor from 0 to 4)
            nib.save(masker.inverse_transform(r2train_dropped), 
             'train_dropping_{}_{:03}.nii'.format(reg,subject))

            nib.save(masker.inverse_transform(r2test_dropped), 
             'test_dropping_{}_{:03}.nii'.format(reg,subject))

            # Compute the difference and save 
            r2train_difference = r2train - r2train_dropped
            nib.save(masker.inverse_transform(r2train_difference), 
             'train_r2_increase_when_adding_{}_{:03}.nii'.format(reg,subject))
            
            r2test_difference = r2test - r2test_dropped
            nib.save(masker.inverse_transform(r2test_difference), 
             'test_r2_increase_when_adding_{}_{:03}.nii'.format(reg,subject))
             
                    
        ### Plot glass brain of the final result (test)
       
            # img = mean_img(masker.inverse_transform(r2test_embeddings_only))
            img=mean_img(masker.inverse_transform(r2test_difference))
            #img=mean_img(thresholded_score_map_img)
            
            display = None
            #display = plot_glass_brain(img, display_mode='lzry', threshold=3.1, colorbar=True, title='{} for alpha = {}, subject{:03}'.format(reg,i,n))
            display = plot_glass_brain(img, display_mode='lzry', colorbar=True, title='Variance explained by {}, subject{:03}'.format(reg,subject))
            
            display.savefig("test_r2_increase_when_adding_{:02}_sub_{:03}.png".format(reg,subject))
            display.close()
                 
                       
            # Save embeddings test
            
             # model_name = r2maps_GLM_crossval_mf_TD2
            #nib.save(masker.inverse_transform(r2test_difference), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/ {}".format(model_name), "{}_test_r2_increase_sub_{:03}.nii".format(reg, subject)))
            nib.save(masker.inverse_transform(r2test_difference), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/r2maps_GLM_crossval_mf_BU2", "{}_test_r2_increase_sub_{:03}.nii".format(reg, subject)))
                 
# """                                         
"""        
       
        img=mean_img(thresholded_score_map_img)
                 display = None
                 display = plot_glass_brain(img, display_mode='lzry', threshold=3.1, colorbar=True, title='{} for alpha = {}, subject{:03}'.format(reg,i,n))
                 display.savefig('alpha{}_{}_only_test_{:03}.png'.format(i,reg,n))
                 display.close()
              
"""  

             
"""                  
# Plot glass brain of the final result (embeddings test)
        img = mean_img(masker.inverse_transform(r2test_embeddings_only))
        display = None
        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, title='Variance explained by word embeddings')
        display.savefig(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_embeddings_only_test_sub_{:03}.png".format(model, alpha, subject)))
        display.close()

# Save embeddings test
        nib.save(masker.inverse_transform(r2test_embeddings_only), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_embeddings_only_test_sub_{:03}.nii".format(model, alpha, subject)))
  
"""
