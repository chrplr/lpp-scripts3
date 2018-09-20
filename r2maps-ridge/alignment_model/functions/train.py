import csv
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFECV

from functions import plot



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
	# Set parameters to unreal values to not raise errors
		alpha = default_alpha
		r2_train_average, r2_val_average = -1, -1

	# Define and train the model
	model = Ridge(alpha=alpha).fit(predictors, data)
	
	# Get r_squared error	
	r2_train_cv = r2_score(data, model.predict(predictors), multioutput='raw_values')
	
	return model, alpha, r2_train_average, r2_val_average, r2_train_cv

def compute_nested_crossval_ridge(predictors, data, groups, alphas):
	"""
	Input : Dictionnary containing all the data of a given subject
	Compute a nested Ridge cross validation.
	Output : Optimal value for alpha for a given voxel
	"""
	# Create a list of alphas from magnitudes
	alphas = np.logspace(alphas[0], alphas[1], alphas[2])

	# Define lengths of the dimensions of the matrix where data will be stored
	n_alphas, n_splits, n_voxels = len(alphas), len(np.unique(groups)), data.shape[1]

	# Define placeholders for R-Squared values
	r2_train, r2_val = np.zeros([n_splits, n_alphas, n_voxels]), np.zeros([n_splits, n_alphas, n_voxels])

	# Manually perform crossvalidation with LeaveOneGroupOut for efficiency
	logo = LeaveOneGroupOut()
	for n_split, (train, val) in enumerate(logo.split(predictors, data, groups=groups)):
		for n_alpha, alpha in enumerate(alphas):
			model = Ridge(alpha=alpha)
			model.fit(predictors[train], data[train])
			
			# Save R-squared errors in the matrix
			r2_train[n_split, n_alpha,] = r2_score(data[train], model.predict(predictors[train]), multioutput='raw_values')
			r2_val[n_split, n_alpha,] = r2_score(data[val], model.predict(predictors[val]), multioutput='raw_values')

	# Average errors over splits
	r2_train_average = np.mean(r2_train, axis=0)
	r2_val_average = np.mean(r2_val, axis=0)

	# Search for the best R-squared errors and the corresponding alphas. Make a list of best_alphas
	r2_max = np.amax(r2_val_average, axis=0)
	r2_max_indices = np.argmax(r2_val_average, axis=0)
	best_alphas = [alphas[i] for i in r2_max_indices]

	return best_alphas, r2_max, r2_train_average, r2_val_average

def compute_PCA(design_matrix, subject, name_ROI, n_components):
	"""
	Input : Design matrix
	Perform PCA on the design matrix to reduce the number of features 
	Output : Transformed design matrix
	"""

	# Save structure (nb_scans)
	len_matrix = [len(i) for i in design_matrix]

	# Stack all the data
	design_matrix = np.vstack(design_matrix)

	# Plot the percentage explained by each PCA component to better choose number of components after
	design_matrix_plot = design_matrix
	pca = PCA(n_components=design_matrix.shape[1])
	pca = pca.fit(design_matrix)
	design_matrix_plot = pca.transform(design_matrix_plot)
	plot.plot_pca(design_matrix.shape[1], pca.explained_variance_ratio_, subject, name_ROI)
	
	# Compute PCA
	pca = PCA(n_components=n_components)
	pca = pca.fit(design_matrix)
	design_matrix = pca.transform(design_matrix)
	
	# Split the matrix in blocks knowing the structure
	matrices = []
	pointer = 0
	for i, length in enumerate(len_matrix):
		matrices.append(design_matrix[pointer:pointer + length])
		pointer = pointer + length
		
	return matrices

def shuffle_dmtx(design_matrix):
	len_matrix = [len(i) for i in design_matrix]
	design_matrix = np.vstack(design_matrix)
	np.transpose(np.random.shuffle(np.transpose(design_matrix)))
	matrices = []
	pointer = 0
	for i, length in enumerate(len_matrix):
		matrices.append(design_matrix[pointer:pointer + length])
		pointer = pointer + length
	
	return matrices
def evaluate_model(X_test, y_test, model):
	"""
	Input : Test data
	Compute R-squared test score with the test data in order to evaluate the model
	Output : R-squared test score
	"""
	return r2_score(y_test, model.predict(X_test), multioutput='raw_values')

def return_predict(X_test, y_test, model):
	prediction = model.predict(X_test)
	
	return prediction, r2_score(y_test, prediction, multioutput='raw_values')

