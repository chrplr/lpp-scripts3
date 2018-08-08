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
	
	# Extact data form data dictionnary
	# data_current = data[:, voxel:voxel +1]

	# Compute nested crossvalidation
	if nested_cv:
		voxels_alpha_train_testcv = {}
		alpha = []
		for voxel in tqdm(range(data.shape[1])):
			alpha_v = compute_nested_crossval_ridge(predictors, data, subject, groups, voxel)
			alpha.append(alpha_v)
	else:
		alpha = pref.defaut_alpha
	r2_train_ncv, r2_val, modeln = -1, -1, -1

	# Training
	model = Ridge(alpha=alpha).fit(predictors, data)

	# Get r_squared error	
	r2_train_cv = r2_score(data, model.predict(predictors), multioutput='raw_values')
	
	return model, [alpha, r2_train_ncv, r2_val, r2_train_cv]



def compute_nested_crossval_ridge(predictors, data, subject, groups, voxel):
	"""
	Input : Dictionnary containing all the data for a given subject
	Extract the data from the dictionnary into 4 variables : X_train, y_train, x_val and y_val. Compute a nested Ridge cross validation.
	Output : Optimal value for alpha for a given voxel
	"""
	
	data_current = data[:, voxel:voxel+1]

	# Training
	alpha_n = []
	r2_test_ncv = []
	logo = LeaveOneGroupOut()
	for train, test in logo.split(predictors, data, groups=groups):
		X_train, X_test = predictors[train], predictors[test]
		y_train, y_test = data_current[train], data_current[test]
		reg = RidgeCV(alphas=pref.alphas_nested_ridgecv, cv=None, store_cv_values=True)
		model = reg.fit(X_train, y_train)
		alpha_n.append(model.alpha_)
	alpha_n = np.mean(np.array(alpha_n))

	# train_ncv_score = r2_score(y_train, model.predict(X_train), multioutput='raw_values')
	
	return np.mean(np.array(alpha_n))

def compute_PCA(predictors, data, X_test):
	"""
	Input : Design matrix
	Perform PCA on the design matrix to reduce the number of features 
	Output : Transformed design matrix
	"""
	
	feat_out_pred, feat_out_test = predictors[:, -5:], X_test[:, -5:]
	X = np.vstack((predictors[:, :-5], X_test[:, :-5]))
	
	pca = PCA(n_components=pref.n_components)
	pca = pca.fit(X)
	
	return np.hstack((pca.transform(predictors[:, :-5]), feat_out_pred)), np.hstack((pca.transform(X_test[:, :-5]), feat_out_test))

def evaluate_model(X_test, y_test, model):
	"""
	Input : Test data
	Compute R-squared test score with the test data in order to evaluate the model
	Output : R-squared test score
	"""
	# return r2_score(y_test, model.predict(X_test))
	return r2_score(y_test, model.predict(X_test), multioutput='raw_values')


def GridSearchCV():
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
	
	pass
