import os
import glob
import csv
from tqdm import tqdm

import numpy as np
import pandas as pd 
import pickle

from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import math_img, mean_img

from sklearn.preprocessing import StandardScaler

from functions import settings_params_preferences as spp
from functions import train
from functions import plot
settings = spp.settings()
params = spp.params()
pref = spp.preferences()


fmri_runs = []
masks = []
ROI = ['IFGorb', 'IFGtri', 'TP', 'TPJ', 'aSTS', 'pSTS']


def get_dmtx_features(features):
	matrices = []
	for block in tqdm(range(params.nb_blocks), unit='block', total=params.nb_blocks):
		features = [''.join(features[i]) for i in range(len(features))]
		features_paths = np.unique(np.hstack([sorted(glob.glob(os.path.join(settings.path2Output, 'all_regressors', 'Block{}'.format(block +1), features[feat]))) for feat in range(len(features))]))
		if not features_paths.shape[0] == 1:
			design_matrice_block = np.array(pd.concat([pd.read_csv(f, header=None) for f in features_paths], axis =1))
		else:
			design_matrice_block = np.array(pd.read_csv(features_paths[0], header=None))	
		matrices.append(design_matrice_block)
	return matrices



def concat_matrices(dmtx_pca, dmtx_no_pca):
	"""
	Concatenate pca and non_pca features in an unique design matrice
	"""
	matrices = [np.hstack((dmtx_pca[i], dmtx_no_pca[i])) for i in range(len(dmtx_pca))]
	
	return matrices


def generate_fmri_data_for_subject(subject, current_ROI):
	"""
	Input : Take as input each fmri file. One file = One block
	Load all fmri data and apply a global mask mak on it. The global mask is computed using the mask from each fmri run (block). 
	Applying a global mask for a subject uniformize the data. 
	Output: Output fmri_runs for a subject, corrected using a global mask
	"""
	fmri_filenames = sorted(glob.glob(os.path.join(settings.rootpath, 
												"fmri-data/en",
												"sub-%03d" % subject, 
												"func", 
												"resample*.nii")))

	masks_filenames = sorted(glob.glob(os.path.join(settings.path2Data,
												"en/fmri_data/masks",
												"sub_{}".format(subject),  
												"resample*.pkl")))

	if current_ROI != -1 and current_ROI <= 5:
		masks_ROIs_filenames = sorted(glob.glob(os.path.join(settings.path2Data, 
												"en/ROIs_masks/",
												"*.nii")))
		ROI_mask = masks_ROIs_filenames[current_ROI]
		ROI_mask = NiftiMasker(ROI_mask, detrend=True, standardize=True)
		ROI_mask.fit()
		fmri_runs = [ROI_mask.transform(f) for f in fmri_filenames]

	else:
		for file in masks_filenames:
			with open(file, 'rb') as f:
				mask = pickle.load(f)
				masks.append(mask)

		global_mask = math_img('img>0.5', img=mean_img(masks))
		masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
		masker.fit()
		fmri_runs = [masker.transform(f) for f in tqdm(fmri_filenames)]
	return fmri_runs
			

def split_data(split, X, y):
	"""
	Input : Take n_long matrices
	Output : Split a matrix data into two matrices of dim n-1 and 1.
	"""
	X_train = [mat for i, mat in enumerate(X) if i!=split]
	X_test = [mat for i, mat in enumerate(X) if i==split]
	y_train = [fmri for i, fmri in enumerate(y) if i!=split]
	y_test = [fmri for i, fmri in enumerate(y) if i==split]
	
	return X_train, X_test, y_train, y_test


def split_train_test_data(current_block, block):
	X_train, X_test, y_train, y_test = [np.array(current_block[i]) for i in range(len(current_block))]

	# Generate group list
	fmri_nb_scans = [len(X_train[i]) for i in range(len(X_train))]
	groups = np.repeat(list(range(8)), fmri_nb_scans)
	
	# Preprocess data for crossvalidation
	predictors = np.vstack(X_train)	#(num_scans_8_blocks, num_features)
	data = np.vstack(y_train)		#(num_scans_8_blocks, num_voxels)
	X_test = np.vstack(X_test)      #(num_scans_1_blocks, num_features)
	y_test = np.vstack(y_test)      #(num_scans_1_blocks, num_voxels)
	
	standardized_scale = StandardScaler().fit(predictors)
	predictors = standardized_scale.transform(predictors)

	standardized_scale = StandardScaler().fit(X_test)
	X_test = standardized_scale.transform(X_test)

	return predictors, data, X_test, y_test, groups


def generate_data_subject(subject, current_ROI, name_ROI, feat_pca, feat_not_pca, hash_):
	"""
	Input : Subject number
	Load fmri data for a subject by first getting design matrices, generate masked fmri data and save data as a pickle.
	Output : Fmri data for a subject in a pickle
	"""
	if not os.path.exists(os.path.join(settings.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{0}_{1}.pkl'.format(subject, hash_))):
		if  feat_pca != None:
			feat_pca = get_dmtx_features(feat_pca)
			matrices_pca = train.compute_PCA(feat_pca, subject, name_ROI)
			matrices = matrices_pca
		
		if  feat_not_pca != None:
			matrices_not_pca = get_dmtx_features(feat_not_pca)
			matrices = matrices_not_pca
		
		if feat_pca != None and feat_not_pca != None:
			matrices = concat_matrices(matrices_pca, matrices_not_pca)
		
		fmri_runs = generate_fmri_data_for_subject(subject, current_ROI)
		gen_data = {'matrices' : matrices, 'fmri_runs' : fmri_runs}

		
		# Save design matrices and fmri data as a dictionnary inside a Pickle
		if not os.path.exists(os.path.join(settings.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI)):
			os.makedirs(os.path.join(settings.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI))
		with open(os.path.join(settings.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{0}_{1}.pkl'.format(subject, hash_)), 'wb') as fout:
				pickle.dump(gen_data, fout)


def get_data(subject, current_ROI, name_ROI, hash_):
	"""
	Load the pickle where the data is stored for a given subject
	"""
	with open(os.path.join(settings.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{0}_{1}.pkl'.format(subject, hash_)), 'rb') as f:
		loaded_data = pickle.load(f)
	matrices = loaded_data['matrices']
	fmri_runs = loaded_data['fmri_runs']
	
	data = []
	for split in range(params.nb_blocks):
		X_train, X_test, y_train, y_test = split_data(split, matrices, fmri_runs)
		data.append([X_train, X_test, y_train, y_test])

	return data
