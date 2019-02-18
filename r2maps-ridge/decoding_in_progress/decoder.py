
################ FILE IN PROGRESS


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

from functions import data as dt 
from functions import train
from functions import settings
from functions import plot


from sklearn.preprocessing import StandardScaler
paths = settings.settings()

nb_blocks = 9 

fmri_runs = dt.generate_fmri_data_for_subject(101, -1)
word_embeddings = dt.get_word_embeddings(101)

data = []
for split in range(nb_blocks):
	X_train, X_test, y_train, y_test = dt.split_data(split, fmri_runs, word_embeddings)
	data.append([X_train, X_test, y_train, y_test])

for block, current_block in tqdm(enumerate(data), unit='block', total=len(data)): 
	predictors, data, X_test, y_test, groups = dt.split_train_test_data(current_block, block)
	model, alpha, r2_train_average, r2_val_average, r2_train_cv = train.train_model_with_nested_cv(predictors, data, 57, groups, False, [0, 6, 20] , 15)
	prediction, r2_test = train.return_predict(X_test, y_test, model)

	prediction_word_emb = np.mean(prediction, axis=1)
	print(prediction_word_emb.shape)
	print(prediction_word_emb)

