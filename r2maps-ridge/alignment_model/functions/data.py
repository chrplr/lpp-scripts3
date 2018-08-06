import os
import glob
import csv

import numpy as np
import pandas as pd 
import pickle
# from tqdm import tqdm

from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import math_img, mean_img

from sklearn.preprocessing import StandardScaler

from functions import settings_params_preferences as spp
settings = spp.settings()
params = spp.params()
pref = spp.preferences()


fmri_runs = []
masks = []
ROI = ['IFGorb', 'IFGtri', 'TP', 'TPJ', 'aSTS', 'pSTS']

def generate_design_matrices(std=False):
	"""
	Input : Individual regressors files for each feature for each block
	Create one unique csv file of dimension sample x features per block.
	Output : Design matrice per block
	"""
	
	for block in range(params.nb_blocks):
		filenames = []
		for feature in range(nb_features):

			filenames.append(os.path.join(settings.path2Output, 'regressors/en', 'Block{0}/{1}_TimeFeat_{2}_reg.csv'.format(block + 1, block + 1, feature + 1)))

		design_matrice_block = pd.concat([pd.read_csv(f) for f in filenames], axis =1)
		design_matrice_block.to_csv(os.path.join(settings.path2Output,'design_matrices/en', 'design_matrice_block{}.csv'.format(block + 1)), index=False)

def get_design_matrices():
	"""
	Input : Design Matrice per block
	Load all design matrices for each block and store them in matrices. Add features and const.
	Output : One giant Design matrice
	"""
	matrices = []
	for block in range(1, 10):
		dmtx = pd.read_csv(os.path.join(settings.path2Output, 'design_matrices/en', 'design_matrice_block{}.csv'.format(block)), header=None)
		
		added_features = pd.read_csv(os.path.join(settings.rootpath, '../MRI/lpp-scripts3/outputs/design-matrices/en/42B_en', 'dmtx{}.csv'.format(block)), header=None)
		added_features = added_features[1:]
		dmtx = np.hstack((dmtx, added_features))

		const = np.ones((dmtx.shape[0], 1))
		dmtx = np.hstack((dmtx, const))
		
		if params.features_of_interest != -1:
			sub_dmtx = [[0]*len(params.features_of_interest)]*len(dmtx)
			for i in range(len(dmtx)):
				sub_dmtx[i] = dmtx[i][params.features_of_interest]
		else:
			sub_dmtx = dmtx
		
		matrices.append(sub_dmtx)
	return matrices


def generate_fmri_data_for_subject(subject, current_ROI):
	"""
	Input : Take as input each fmri file. One file = One block
	Load all fmri data and apply a global mask mak on it. The global mask is computed using the mask from each fmri run (block). 
	Applying a global mask for a subject uniformize the data. 
	Output: Output fmri_runs for a subject, corrected using a global mask
	"""
	fmri_filenames = sorted(glob.glob(os.path.join(settings.rootpath, 
												"../MRI/fmri-data/en",
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
		#fmri_runs = [ROI_mask.transform(f) for f in tqdm(fmri_filenames)]
		fmri_runs = [ROI_mask.transform(f) for f in fmri_filenames]

	else:
		for file in masks_filenames:
			with open(file, 'rb') as f:
				mask = pickle.load(f)
				masks.append(mask)

		global_mask = math_img('img>0.5', img=mean_img(masks))
		masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
		masker.fit()
		# fmri_runs = [masker.transform(f) for f in tqdm(fmri_filenames)]
		fmri_runs = [masker.transform(f) for f in fmri_filenames]

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
	X_train = np.array(current_block[0])
	X_test = np.array(current_block[1])
	y_train = np.array(current_block[2])
	y_test = np.array(current_block[3])

	# Preprocess data for crossvalidation
	predictors = np.vstack(X_train)	#(num_scans_8_blocks, num_features)
	data = np.vstack(y_train)		#(num_scans_8_blocks, num_voxels)
	X_test = np.vstack(X_test)      #(num_scans_1_blocks, num_features)
	y_test = np.vstack(y_test)      #(num_scans_1_blocks, num_voxels)
	
	standardized_scale = StandardScaler().fit(predictors)
	predictors = standardized_scale.transform(predictors)

	standardized_scale = StandardScaler().fit(X_test)
	X_test = standardized_scale.transform(X_test)

	# Define groups for voxels
	groups = []
	for j in range(params.nb_blocks):
		if j != block:
			groups = groups + [j + 1]*params.scans[j]

	return predictors, data, X_test, y_test, groups


def generate_data_all_subjects(subject, current_ROI):
	"""
	Input : Subject number
	Load fmri data for a subject by first getting design matrices, generate masked fmri data and save data as a pickle.
	Output : Fmri data for a subject in a pickle
	"""
	# Generate design matrices and fmri data
	matrices = get_design_matrices()
	fmri_runs = generate_fmri_data_for_subject(subject, current_ROI)
	gen_data = {'matrices' : matrices, 'fmri_runs' : fmri_runs}
	
	#Get ROI name
	name_ROI = ROI[current_ROI]
	if current_ROI == -1: name_ROI = 'All'
	
	# Save design matrices and fmri data as a dictionnary inside a Pickle
	if not os.path.exists(os.path.join(settings.path2Data, 'en/crossval_splits_added_features', 'Sub_{}'.format(subject), name_ROI)):
		os.makedirs(os.path.join(settings.path2Data, 'en/crossval_splits_added_features', 'Sub_{}'.format(subject), name_ROI))
	with open(os.path.join(settings.path2Data, 'en/crossval_splits_added_features', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{}.pkl'.format(subject)), 'wb') as fout:
			pickle.dump(gen_data, fout)


def get_data(subject, current_ROI):
	"""
	Load the pickle where the data is stored for a given subject
	"""
	data = []
	
	# Get ROi name 
	name_ROI = ROI[current_ROI]
	if current_ROI == -1: name_ROI = 'All'

	# Load data 
	with open(os.path.join(settings.path2Data, 'en/crossval_splits_added_features', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{}.pkl'.format(subject)), 'rb') as f:
		loaded_data = pickle.load(f)
	matrices = loaded_data['matrices']
	fmri_runs = loaded_data['fmri_runs']
	
	# for split in tqdm(range(params.nb_blocks)):
	for split in range(params.nb_blocks):
		X_train, X_test, y_train, y_test = split_data(split, matrices, fmri_runs)
		data.append([X_train, X_test, y_train, y_test])

	return data


