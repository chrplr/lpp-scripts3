import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import math_img, mean_img

from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFECV


from sklearn.preprocessing import StandardScaler

import paths
paths = paths.paths()

nb_blocks = 9 
def generate_fmri_data_for_subject(subject):
	"""
	Input : Take as input each fmri file. One file = One block
	Load all fmri data and apply a global mask mak on it. The global mask is computed using the mask from each fmri run (block). 
	Applying a global mask for a subject uniformize the data. 
	Output: Output fmri_runs for a subject, corrected using a global mask
	"""
	fmri_filenames = sorted(glob.glob(os.path.join(paths.rootpath, 
												"fmri-data/en",
												"sub-%03d" % subject, 
												"func", 
												"resample*.nii")))
	

	masks_filenames = sorted(glob.glob(os.path.join(paths.path2Data,
												"en/fmri_data/masks",
												"sub_{}".format(subject),  
												"resample*.pkl")))
	masks = []
	for file in masks_filenames:
		with open(file, 'rb') as f:
			mask = pickle.load(f)
			masks.append(mask)

	global_mask = math_img('img>0.5', img=mean_img(masks))
	masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
	masker.fit()
	fmri_runs = [masker.transform(f) for f in tqdm(fmri_filenames)]
	print(fmri_runs[0].shape)
		
	return fmri_runs

def get_word_embeddings(subject):
	dmtx = []
	for block in range(1, 10):
		word_emb_filenames = sorted(glob.glob(os.path.join(paths.path2Data, 
													"../all_regressors",
													"Block{}".format(block), 
													"word_emb*.csv")))
		
		dmtx_block = np.array(pd.concat([pd.read_csv(f, header=None) for f in word_emb_filenames], axis =1))
		dmtx.append(dmtx_block)

	return dmtx


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


def train_model_with_nested_cv(predictors, data, subject, groups, nested_cv, alphas, default_alpha):
	"""
	Input : Alpha from the Ridge crossvalidation. 
	Perform Ridge crossvalidation on data. 
	Output : r2 score of the training set and model 
	"""      

	# Compute nested crossvalidation
	if nested_cv:
		alpha, r2_max, r2_train_average, r2_val_average = compute_nested_crossval_ridge(predictors, data, groups, alphas)
	else:
		alpha = default_alpha
		r2_train_average, r2_val_average = -1, -1

	# Training
	model = Ridge(alpha=alpha).fit(predictors, data)
	# Get r_squared error	
	r2_train_cv = r2_score(data, model.predict(predictors), multioutput='raw_values')
	
	return model, alpha, r2_train_average, r2_val_average, r2_train_cv



def compute_nested_crossval_ridge(predictors, data, groups, alphas):
	"""
	Input : Dictionnary containing all the data for a given subject
	Extract the data from the dictionnary into 4 variables : X_train, y_train, x_val and y_val. Compute a nested Ridge cross validation.
	Output : Optimal value for alpha for a given voxel
	"""
	alphas = np.logspace(alphas[0], alphas[1], alphas[2])

	n_alphas, n_splits, n_voxels = len(alphas), len(np.unique(groups)), data.shape[1]
	r2_train, r2_val = np.zeros([n_splits, n_alphas, n_voxels]), np.zeros([n_splits, n_alphas, n_voxels])

	logo = LeaveOneGroupOut()
	for n_split, (train, val) in enumerate(logo.split(predictors, data, groups=groups)):
		for n_alpha, alpha in enumerate(alphas):
			model = Ridge(alpha=alpha)
			model.fit(predictors[train], data[train])
			
			r2_train[n_split, n_alpha,] = r2_score(data[train], model.predict(predictors[train]), multioutput='raw_values')
			r2_val[n_split, n_alpha,] = r2_score(data[val], model.predict(predictors[val]), multioutput='raw_values')

	r2_train_average = np.mean(r2_train, axis=0)
	r2_val_average = np.mean(r2_val, axis=0)

	r2_max = np.amax(r2_val_average, axis=0)
	r2_max_indices = np.argmax(r2_val_average, axis=0)
	best_alphas = [alphas[i] for i in r2_max_indices]

	return best_alphas, r2_max, r2_train_average, r2_val_average

def evaluate_model(X_test, y_test, model):
	"""
	Input : Test data
	Compute R-squared test score with the test data in order to evaluate the model
	Output : R-squared test score
	"""
	return r2_score(y_test, model.predict(X_test), multioutput='raw_values')


fmri_runs = generate_fmri_data_for_subject(101)
word_embeddings = get_word_embeddings(101)

data = []
for split in range(nb_blocks):
	X_train, X_test, y_train, y_test = split_data(split, fmri_runs, word_embeddings)
	data.append([X_train, X_test, y_train, y_test])

for block, current_block in tqdm(enumerate(data), unit='block', total=len(data)): 
	predictors, data, X_test, y_test, groups = split_train_test_data(current_block, block)
	model, alpha, r2_train_average, r2_val_average, r2_train_cv = train_model_with_nested_cv(predictors, data, 57, groups, False, None , 15)
	r2_test = evaluate_model(X_test, y_test, model)
	
	print(r2_test)

