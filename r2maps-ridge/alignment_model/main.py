import os
import pickle 
import argparse
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from functions import data as dt
from functions import train
from functions import plot
from functions import settings
paths = settings.paths()

import matplotlib.pyplot as plt

TEST_P_VALUE = False
model_p_values = 2

# Parse arguments
parser = argparse.ArgumentParser(description='Parameters for computing')

parser.add_argument('--subject', '-s', type=int, default=57,
					help='Subject flag')
parser.add_argument('--ROI', '-r', type=int, default=-1,
					help='ROI to consider. If -1, all brain is considered')
parser.add_argument('--nested', '-n', action='store_true',
					help='Perform nested_crossvalidation')
parser.add_argument('--plot_nested', '-pn', action='store_true',
					help='Plot train and test nested crossvalidation R-squared errors depending on alpha ')
parser.add_argument('--file', '-f', default='no_filename',
					help='Name of the glass brain figure')
parser.add_argument('--pca', '-p', type=list, nargs='+',
					help='Features in the model that need to go through a PCA')
parser.add_argument('--n_components', '-c', type=int, default=51,
					help='Number of coponents for the PCA')
parser.add_argument('--features_out', '-fo', type=list, nargs='+',
					help='Features in the model whiwh do not need to go througn a PCA')
parser.add_argument('--alphas', '-a', type=list, nargs=3, default=[0, 6, 20],
					help='min power of ten, max power of ten, number of alphas. Example : 0, 6, 20 means 20 alpha values logspaced between 10^0 and 10^6')
parser.add_argument('--default_alpha', '-da', type=int, default=15,
					help='Set default alpha value for non-nested computations')
parser.add_argument('--shuffle', type=int, default=None,
					help='Generate p-value mask for a given experiment')
parser.add_argument('--p_value', '-pv', default=None,
					help='Path to all matrices shuffled')


args = parser.parse_args()

# Generate a hash specific to arguments given. Used to retrieve data if data is already generated
params_to_hash = str(args.subject) + str(args.nested) + str(args.ROI) + str(args.pca) + str(args.features_out)
hash_ = hashlib.sha1(params_to_hash.encode()).hexdigest()

# Get the name of the ROI
name_ROI = ['IFGorb', 'IFGtri', 'TP', 'TPJ', 'aSTS', 'pSTS', 'All'][args.ROI]

# Generate the data and save it to a pickle
print('Subject : ', args.subject,'| ROI : ',name_ROI,' | File : ',args.file,' | Hash : ',hash_)
print('Generating data ...')
dt.generate_data_subject(args.subject, args.ROI, name_ROI, args.pca, args.features_out, hash_, args.n_components)

# Load the data from the pickle
print('Loading data ...')
data = dt.get_data(args.subject, args.ROI, name_ROI, hash_, args.shuffle)

# Check outisde of the loop if our model-saving directory exists
if not os.path.exists(os.path.join(paths.path2Output, 'models_blocks', 'Sub_{}'.format(args.subject), '{}'.format(name_ROI))):
	os.makedirs(os.path.join(paths.path2Output, 'models_blocks', 'Sub_{}'.format(args.subject), '{}'.format(name_ROI)))

# Train the model
print('Training ...')
r2_tests, r2_trains = [], []

# Loop over blocks
for block, current_block in tqdm(enumerate(data), unit='block', total=len(data)): 
	# Split, train and evaluate the block
	predictors, data, X_test, y_test, groups = dt.split_train_test_data(current_block, block)
	model, alpha, r2_train_average, r2_val_average, r2_train_cv = train.train_model_with_nested_cv(predictors, data, args.subject, groups, args.nested, args.alphas, args.default_alpha)
	r2_test = train.evaluate_model(X_test, y_test, model)
	
	r2_trains.append(r2_train_cv)
	r2_tests.append(r2_test)

	# Make plots of nested crossvalidation validation values
	if args.plot_nested: plot.regularization_path(r2_train_average, r2_val_average, args.subject, block, name_ROI, args.file, args.alphas)

	# Save model
	pickle.dump(model, open(os.path.join(paths.path2Output,'models_blocks', 'Sub_{}'.format(args.subject), '{}'.format(name_ROI), 'Block_{0}_{1}.pkl'.format(block +1, args.file)), 'wb'))


# Save r2_tests
if args.shuffle != None:
	with open('shuffle/shuffle_{0}_{1}_{2}.pkl'.format(args.shuffle, args.file, args.subject), 'wb') as f:
		pickle.dump(r2_tests, f)

# Clean R2_scores
r2_trains = np.mean(np.vstack(r2_trains), axis=0)
r2_trains[r2_trains < 0], r2_trains[r2_trains > 0.99] = 0, 0 
r2_tests = np.mean(np.vstack(r2_tests), axis=0)
r2_tests[r2_tests < 0], r2_tests[r2_tests > 0.99] = 0, 0
	
#Save r2_test_ no_shuffle
if args.shuffle == None:
	with open('save_r2test/{}_{}.pkl'.format(args.file, args.subject), 'wb') as f:
		pickle.dump(r2_tests, f)

