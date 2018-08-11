import os
import csv
import sys
import glob
import pickle 
import argparse
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from functions import data as dt
from functions import train
from functions import plot
from functions import settings_params_preferences as spp

import matplotlib.pyplot as plt

plt.switch_backend('Agg')


parser = argparse.ArgumentParser(description='Parameters for computing')
parser.add_argument('--subject', '-s', 
					type=int,
					default=57,
					help='Subject flag')
parser.add_argument('--ROI', '-r', 
					type=int, 
					default=-1,
					help='ROI to consider. If -1, all brain is considered')

parser.add_argument('--nested', '-n', 
					action='store_true',
					help='Perform nested_crossvalidation')

parser.add_argument('--plot_nested', '-pn', 
					action='store_true',
					help='Plot train and test nested crossvalidation R-squared errors depending on alpha ')

parser.add_argument('--file', '-f', 
					default='no_filename',
					help='Name of the glass brain figure')

parser.add_argument('--pca', '-p', 
					type=list,
					nargs='+',
					help='Features in the model that need to go through a PCA')

parser.add_argument('--features_out', '-fo', 
					type=list,
					nargs='+',
					help='Features in the model whiwh do not need to go througn a PCA')

args = parser.parse_args()

subject, nested, plot_nested, filename, current_ROI, feat_pca, feat = args.subject, args.nested, args.plot_nested, args.file, args.ROI, args.pca, args.features_out

params_to_hash = str(subject) + str(nested) + str(current_ROI) + str(feat_pca) + str(feat)
hash_ = hashlib.sha1(params_to_hash.encode()).hexdigest()


name_ROI = ['IFGorb', 'IFGtri', 'TP', 'TPJ', 'aSTS', 'pSTS', 'All'][current_ROI]

print('\n')
print('Loading settings ...')
settings = spp.settings()
print('Loading parameters ...')
params = spp.params()
print('Loading preferences ... \n')
pref = spp.preferences()

print('Subject : ', subject)
print('ROI : ', name_ROI)
print('File : ', filename, '\n')
print('Hash : ', hash_, '\n' )


print('Generating data ...')
dt.generate_data_subject(subject, current_ROI, name_ROI, feat_pca, feat, hash_)
print('Loading data ...')
data = dt.get_data(subject, current_ROI, name_ROI, hash_)

if not os.path.exists(os.path.join(settings.path2Output, 'models_blocks', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
	os.makedirs(os.path.join(settings.path2Output, 'models_blocks', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))

r2_tests = []
r2_trains = []
for block, current_block in tqdm(enumerate(data), unit='block', total=len(data)): 
	
	predictors, data, X_test, y_test, groups = dt.split_train_test_data(current_block, block)
	model, alpha, r2_train_average, r2_val_average, r2_train_cv = train.train_model_with_nested_cv(predictors, data, subject, groups, nested)
	r2_test = train.evaluate_model(X_test, y_test, model)
	
	r2_trains.append(r2_train_cv)
	r2_tests.append(r2_test)

	if plot_nested: plot.regularization_path(r2_train_average, r2_val_average, subject, block, name_ROI)

	pickle.dump(model, open(os.path.join(settings.path2Output,'models_blocks', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Block_{}.pkl'.format(block +1)), 'wb'))

# Clean R2scores
r2_trains = np.mean(np.vstack(r2_trains), axis=0)
r2_trains[r2_trains < 0], r2_trains[r2_trains > 0.99] = 0, 0 
r2_tests = np.mean(np.vstack(r2_tests), axis=0)
r2_tests[r2_tests < 0], r2_tests[r2_tests > 0.99] = 0, 0
	
# Plotting
plot.glass_brain(plot.mask_inverse(r2_tests, subject, current_ROI), subject, name_ROI, filename)

