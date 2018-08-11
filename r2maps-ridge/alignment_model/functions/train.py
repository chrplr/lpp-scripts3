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

from functions import settings_params_preferences as spp
from functions import plot
settings = spp.settings()
params = spp.params()
pref = spp.preferences()


def train_model_with_nested_cv(predictors, data, subject, groups, nested_cv):
	"""
	Input : Alpha from the Ridge crossvalidation. 
	Perform Ridge crossvalidation on data. 
	Output : r2 score of the training set and model 
	"""      

	# Compute nested crossvalidation
	if nested_cv:
		alpha, r2_max, r2_train_average, r2_val_average = compute_nested_crossval_ridge(predictors, data, groups)
	else:
		alpha = pref.defaut_alpha
		r2_train_average, r2_val_average = -1, -1

	# Training
	model = Ridge(alpha=alpha, fit_intercept=True).fit(predictors, data)

	# Get r_squared error	
	r2_train_cv = r2_score(data, model.predict(predictors), multioutput='raw_values')
	
	return model, alpha, r2_train_average, r2_val_average, r2_train_cv



def compute_nested_crossval_ridge(predictors, data, groups):
	"""
	Input : Dictionnary containing all the data for a given subject
	Extract the data from the dictionnary into 4 variables : X_train, y_train, x_val and y_val. Compute a nested Ridge cross validation.
	Output : Optimal value for alpha for a given voxel
	"""
	n_alphas = len(pref.alphas_nested_ridgecv)
	n_splits = len(np.unique(groups))
	n_voxels = data.shape[1]
	r2_train, r2_val = np.zeros([n_splits, n_alphas, n_voxels]), np.zeros([n_splits, n_alphas, n_voxels])

	logo = LeaveOneGroupOut()
	for n_split, (train, val) in enumerate(logo.split(predictors, data, groups=groups)):
		for n_alpha, alpha in enumerate(pref.alphas_nested_ridgecv):
			model = Ridge(alpha=alpha, fit_intercept=True)
			model.fit(predictors[train], data[train])
			
			r2_train[n_split, n_alpha,] = r2_score(data[train], model.predict(predictors[train]), multioutput='raw_values')
			r2_val[n_split, n_alpha,] = r2_score(data[val], model.predict(predictors[val]), multioutput='raw_values')

	r2_train_average = np.mean(r2_train, axis=0)
	r2_val_average = np.mean(r2_val, axis=0)

	r2_max = np.amax(r2_val_average, axis=0)
	r2_max_indices = np.argmax(r2_val_average, axis=0)
	best_alphas = [pref.alphas_nested_ridgecv[i] for i in r2_max_indices]

	return best_alphas, r2_max, r2_train_average, r2_val_average

def compute_PCA(design_matrix, subject, name_ROI):
	"""
	Input : Design matrix
	Perform PCA on the design matrix to reduce the number of features 
	Output : Transformed design matrix
	"""

	len_matrix = [len(i) for i in design_matrix]
	design_matrix = np.vstack(design_matrix)
	
	if pref.plot_pca:
		design_matrix_plot = design_matrix
		pca = PCA(n_components=design_matrix.shape[1])
		pca = pca.fit(design_matrix)
		design_matrix_plot = pca.transform(design_matrix_plot)
		plot.plot_pca(design_matrix.shape[1], pca.explained_variance_ratio_, subject, name_ROI)
	
	pca = PCA(n_components=pref.n_components)
	pca = pca.fit(design_matrix)
	design_matrix = pca.transform(design_matrix)
	
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


# def GridSearchCV():
	"""
	Input : Dictionnary containing all the data for a given subject
	Extract the data from the dictionnary into 4 variables : X_train, y_train, x_val and y_val. Compute a nested Ridge cross validation.
	Output : Optimal value for alpha for a given voxel
	"""
	
	# logo = LeaveOneGroupOut()
	# reg = GridSearchCV(Ridge(fit_intercept=params.fit_intercept), param_grid={"alpha": params.alphas_nested_ridgecv}, return_train_score=True, cv=logo, n_jobs=-2)
	# model = reg.fit(predictors, data_current, groups=groups)


	# return model.best_params_['alpha'], model.cv_results_['mean_train_score'] , model.cv_results_['mean_test_score'], model
	# return alpha_n
