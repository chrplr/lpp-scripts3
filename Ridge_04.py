# IMPORTS


import numpy as np
from nilearn.masking import compute_epi_mask
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
from nilearn.image import mean_img, math_img

import glob
import os
import os.path
import nibabel as nib
import csv
import pandas as pd


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut




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
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data = np.vstack([fmri_runs[i] for i in train])
        predictors = np.vstack([design_matrices[i] for i in train])
        model_ridge = Ridge(alpha=0.4).fit(predictors, fmri_data)
            
        rsquares_training = clean_rscores(r2_score(fmri_data, 
                                                   model_ridge.predict(predictors), multioutput='raw_values'), 
                                          .0, .99)
        test_run = test[0]
        rsquares_test = clean_rscores(r2_score(fmri_runs[test_run], 
                                               model_ridge.predict(design_matrices[test_run]), multioutput='raw_values'),
                                      .0, .99)
        
        log(rsquares_training, rsquares_test)

        r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, rsquares_training])    
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, rsquares_test])
        
    return (np.mean(r2_train, axis=0), np.mean(r2_test, axis=0))
    
    

if __name__ == '__main__':
 
    rootdir = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI"
    #rootdir = "/media/sneza/FREECOM HDD/LePetitPrince_Pallier_2018/MRI"

    model = '42B_en'
    matrices = get_design_matrices(rootdir, model)
    alpha = 0.4
    subjects = [71]
 #   subjects = [78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]

#    for num, subject in zip(range(len(subjects)), subjects):
    for subject in subjects:
    
        masker = compute_global_masker(rootdir, [subject])
    
        fmri_filenames = sorted(glob.glob(os.path.join(rootdir, 
                                                   "fmri-data/en",
                                                   "sub-%03d" % subject, 
                                                   "func", 
                                                   "resample*.nii")))
        fmri_runs = [masker.transform(f) for f in fmri_filenames]
    
        loglabel = subject
        logcsvwriter = csv.writer(open(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "new_subjects_stats_{}_alpha_{}.log".format(model, alpha)), "a+"))
        
        

        # Compute and save training and test of the full model
        r2train, r2test = compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)
        nib.save(masker.inverse_transform(r2train), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_train_sub_{:03}.nii".format(model, alpha, subject)))

        nib.save(masker.inverse_transform(r2test), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_test_sub_{:03}.nii".format(model, alpha, subject)))
 
 
        ### Compare the full model with the model where embeddings are dropped
        
        # Drop embeddings 
        without_embeddings = [np.delete(mtx, np.s_[:300], 1) for mtx in matrices]
        r2train_simple, r2test_simple = compute_crossvalidated_r2(fmri_runs, without_embeddings,  loglabel, logcsvwriter)
        
        # Save the simple model (without embeddings)
#        nib.save(masker.inverse_transform(r2test_simple), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_rms_wordrate_freq_bottomup_sub_{:03}.nii".format(model, alpha, subject)))
        
        # Compute the difference and save 
        r2train_embeddings_only = r2train - r2train_simple
        nib.save(masker.inverse_transform(r2train_embeddings_only), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_embeddings_only_train_sub_{:03}.nii".format(model, alpha, subject)))
        r2test_embeddings_only = r2test - r2test_simple
        
        
        # Plot glass brain of the final result (embeddings test)

        img = mean_img(masker.inverse_transform(r2test_embeddings_only))
        display = None
        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, vmax=0.15, title='Sujet 51')
        display.savefig(os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "master_en_{}.png".format(subject)))
        display.close()
        # Save embeddings test
        nib.save(masker.inverse_transform(r2test_embeddings_only), os.path.join(rootdir, "lpp-scripts3/outputs/results-indiv/en/{}".format(model), "{}_alpha_{}_embeddings_only_test_sub_{:03}.nii".format(model, alpha, subject)))
