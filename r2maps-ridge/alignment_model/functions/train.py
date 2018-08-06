import csv
import numpy as np

from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.linear_model import Ridge
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


def train_model_with_nested_cv(predictors, data, subject, voxel, groups, nested_cv=pref.ridge_nested_crossval):
	"""
	Input : Alpha from the Ridge crossvalidation. 
	Perform Ridge crossvalidation on data. 
	Output : r2 score of the training set and model 
	"""       
	
	# Extact data form data dictionnary
	data_current = data[:, voxel:voxel +1]

	# Compute nested crossvalidation
	if nested_cv:
		alpha, r2_train_ncv, r2_val, modeln = compute_nested_crossval_ridge(predictors, data_current, subject, voxel, groups)
	else:
		alpha = params.defaut_alpha
		r2_train_ncv, r2_val, modeln = -1, -1, -1

	# Training
	model = Ridge(alpha=alpha).fit(predictors, data_current)
	

	# print(predictors.shape)
	# print(data_current)
	# model = Ridge(alpha=alpha)
	# selector = RFECV(model, step=1)
	# selector.fit(predictors, data_current)
	# predictors = selector.fit_transform(predictors, data_current)
	# print(predictors.shape)

	# model.fit(predictors, data_current)

	# Get r_squared error	
	#r2_train_cv = r2_score(data_current, model.predict(predictors))
	r2_train_cv = -1		
	
	return model, modeln, [voxel, alpha, r2_train_ncv, r2_val, r2_train_cv]


def compute_nested_crossval_ridge(predictors, data_current, subject, voxel, groups):
	"""
	Input : Dictionnary containing all the data for a given subject
	Extract the data from the dictionnary into 4 variables : X_train, y_train, x_val and y_val. Compute a nested Ridge cross validation.
	Output : Optimal value for alpha for a given voxel
	"""

	# Training
	logo = LeaveOneGroupOut()
	reg = GridSearchCV(Ridge(fit_intercept=params.fit_intercept), param_grid={"alpha": params.alphas_nested_ridgecv}, return_train_score=True, cv=logo, n_jobs=-2)
	model = reg.fit(predictors, data_current, groups=groups)

	return model.best_params_['alpha'], model.cv_results_['mean_train_score'] , model.cv_results_['mean_test_score'], model

def compute_PCA(predictors, data, X_test):
	"""
	Input : Design matrix
	Perform PCA on the design matrix to reduce the number of features 
	Output : Transformed design matrix
	"""
	feat_out_pred = predictors[:, -5:]
	feat_out_test = X_test[:, -5:]
	X = np.vstack((predictors[:, :-5], X_test[:, :-5]))
	pca = PCA(n_components=pref.n_components)
	pca = pca.fit(X)
	
	# PCA PLotting
	# print(len(pca.explained_variance_ratio_))
	# plot.plot_pca(pca.explained_variance_ratio_)

	return np.hstack((pca.transform(predictors[:, :-5]), feat_out_pred)), np.hstack((pca.transform(X_test[:, :-5]), feat_out_test))

def evaluate_model(X_test, y_test, model):
	"""
	Input : Test data
	Compute R-squared test score with the test data in order to evaluate the model
	Output : R-squared test score
	"""
	return r2_score(y_test, model.predict(X_test))


# def feature_selection(predictors, data, X_test):
# 	print("Feature Selection")
# 	X = np.vstack((predictors, X_test))
# 	clf = ExtraTreesClassifier()
# 	clf.fit(predictors, data)
# 	model = SelectFromModel(clf, prefit=True)
# 	X_new = model.transform(X)
# 	print(X_new.shape)