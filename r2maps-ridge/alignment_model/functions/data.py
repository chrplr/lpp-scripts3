import os
import glob
from tqdm import tqdm

import numpy as np
import pandas as pd 
import pickle

from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import math_img, mean_img

from sklearn.preprocessing import StandardScaler

from functions import settings as settings
from functions import train
from functions import plot
paths = settings.paths()

nb_blocks = 9

def get_dmtx_features(features):
	matrices = []
	# Loop over blocks
	for block in tqdm(range(nb_blocks), unit='block', total=nb_blocks):
		# Format the arguments passed through the shell into one string
		features = [''.join(features[i]) for i in range(len(features))]

		# Search for paths corresponding to glob patterns present in features
		features_paths = np.unique(np.hstack([sorted(glob.glob(os.path.join(paths.path2Output, 'all_regressors', 'Block{}'.format(block +1), features[feat]))) for feat in range(len(features))]))
		
		# Concatenate all the data contained vy the file into one matrix
		design_matrice_block = np.array(pd.concat([pd.read_csv(f, header=None) for f in features_paths], axis =1) if features_paths.shape[0] != 1 else pd.read_csv(features_paths[0], header=None))
		matrices.append(design_matrice_block)
	return matrices


def concat_matrices(dmtx_pca, dmtx_no_pca):
	"""
	Concatenate pca and non_pca features in an unique design matrice
	"""
	return [np.hstack((dmtx_pca[i], dmtx_no_pca[i])) for i in range(len(dmtx_pca))]


def generate_fmri_data_for_subject(subject, current_ROI):
	"""
	Input : Take as input each fmri file. One file = One block
	Load all fmri data and apply a global mask mak on it. The global mask is computed using the mask from each fmri run (block). 
	Applying a global mask for a subject uniformize the data. 
	Output: Output fmri_runs for a subject, corrected using a global mask
	"""

	# Get all paths for fmri data
	fmri_filenames = sorted(glob.glob(os.path.join(paths.rootpath, 
												"fmri-data/en",
												"sub-%03d" % subject, 
												"func", 
												"resampled*.nii")))
	
	# Process for All brain
	if current_ROI == -1:
		# Get paths for masks
		masks_filenames = sorted(glob.glob(os.path.join(paths.path2Data,
												"en/fmri_data/masks",
												"sub_{}".format(subject),  
												"resample*.pkl")))
		masks = []
		for file in masks_filenames:
			with open(file, 'rb') as f:
				mask = pickle.load(f)
				masks.append(mask)

		# Compute a global mask for all subject. This way the data will be uniform
		global_mask = math_img('img>0.5', img=mean_img(masks))
		masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
		masker.fit()

		# Apply the mask to each fmri run (block)
		fmri_runs = [masker.transform(f) for f in tqdm(fmri_filenames)]
	
	# Process for a  specific ROI
	else:
		# get paths of ROIs masks
		masks_ROIs_filenames = sorted(glob.glob(os.path.join(paths.path2Data, 
												"en/ROIs_masks/",
												"*.nii")))
		# Choose the mask 
		ROI_mask = masks_ROIs_filenames[current_ROI]
		ROI_mask = NiftiMasker(ROI_mask, detrend=True, standardize=True)
		ROI_mask.fit()

		# Apply the mask to each fmri run (block)
		fmri_runs = [ROI_mask.transform(f) for f in fmri_filenames]
		
	return fmri_runs
			

def split_data(split, X, y, matrices_shuffled):
	"""
	Input : Take n_long matrices
	Output : Split a matrix data into two matrices of dim n-1 and 1.
	"""
	X_train = [mat for i, mat in enumerate(X) if i!=split]
	X_test = [mat for i, mat in enumerate(X) if i==split]
	y_train = [fmri for i, fmri in enumerate(y) if i!=split]
	y_test = [fmri for i, fmri in enumerate(y) if i==split]
	
	if matrices_shuffled != None: X_train = [mat for i, mat in enumerate(matrices_shuffled) if i!=split]
	
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
	
	# Standardize design matrices
	standardized_scale = StandardScaler().fit(predictors)
	predictors = standardized_scale.transform(predictors)
	standardized_scale = StandardScaler().fit(X_test)
	X_test = standardized_scale.transform(X_test)

	return predictors, data, X_test, y_test, groups


def generate_data_subject(subject, current_ROI, name_ROI, feat_pca, feat_not_pca, hash_, n_components):
	"""
	Input : Subject number
	Load fmri data for a subject by first getting design matrices, generate masked fmri data and save data as a pickle.
	Output : Fmri data for a subject in a pickle
	"""

	# Verify iof the data has already been generated
	if not os.path.exists(os.path.join(paths.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{0}_{1}.pkl'.format(subject, hash_))):
		if  feat_pca != None:
			feat_pca = get_dmtx_features(feat_pca)
			matrices_pca = train.compute_PCA(feat_pca, subject, name_ROI, n_components)
			matrices = matrices_pca
		
		if  feat_not_pca != None:
			matrices_not_pca = get_dmtx_features(feat_not_pca)
			matrices = matrices_not_pca
			if feat_pca != None:
				matrices = concat_matrices(matrices_pca, matrices_not_pca)
		
		fmri_runs = generate_fmri_data_for_subject(subject, current_ROI)
		gen_data = {'matrices' : matrices, 'fmri_runs' : fmri_runs}

		
		# Save design matrices and fmri data as a dictionnary inside a Pickle
		if not os.path.exists(os.path.join(paths.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI)):
			os.makedirs(os.path.join(paths.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI))
		with open(os.path.join(paths.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{0}_{1}.pkl'.format(subject, hash_)), 'wb') as fout:
				pickle.dump(gen_data, fout)


def get_data(subject, current_ROI, name_ROI, hash_, shuffle):
	"""
	Load the pickle where the data is stored for a given subject
	"""
	with open(os.path.join(paths.path2Data, 'en/data_subjects_fmri_dmtx', 'Sub_{}'.format(subject), name_ROI, 'data_sub_{0}_{1}.pkl'.format(subject, hash_)), 'rb') as f:
		loaded_data = pickle.load(f)
	
	matrices = loaded_data['matrices']
	fmri_runs = loaded_data['fmri_runs']
	
	# Shuffle matrices for p_value
	if shuffle != None: matrices_shuffled = train.shuffle_dmtx(matrices)
	else: matrices_shuffled = None
	# Split the pickle and make a list of usable train and test data
	data = []
	for split in range(nb_blocks):
		X_train, X_test, y_train, y_test = split_data(split, matrices, fmri_runs, matrices_shuffled)
		data.append([X_train, X_test, y_train, y_test])

	return data

###################################################################################################
#Test for Decoding
###################################################################################################

def get_word_embeddings(subject):
	dmtx = []
	for block in range(1, 10):
		# Get paths for word embeddings
		word_emb_filenames = sorted(glob.glob(os.path.join(paths.path2Data, 
													"../all_regressors",
													"Block{}".format(block), 
													"word_emb*.csv")))
		
		# Build a design matrix
		dmtx_block = np.array(pd.concat([pd.read_csv(f, header=None) for f in word_emb_filenames], axis =1))
		dmtx.append(dmtx_block)

	return dmtx

