import os
import csv
import sys
import pickle 
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from functions import data as dt
from functions import train
from functions import plot
from functions import settings_params_preferences as spp

import matplotlib.pyplot as plt

from joblib import Memory
from tempfile import mkdtemp
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)


#Parser
#----------------------------------------------------------------------
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

parser.add_argument('--test', '-t', 
					action='store_true',
					default=False,
					help='Test pipeline on 20 voxels')

parser.add_argument('--file', '-f', 
					default='no_filename',
					help='Name of the glass brain figure')

args = parser.parse_args()
#-----------------------------------------------------------------------

subject, nested, filename, current_ROI = args.subject, args.nested, args.file, args.ROI
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

plt.switch_backend('Agg')



# Compute crossvalidation for each subject

print('Generating data ...')
if pref.generate_data: memory.cache(dt.generate_data_all_subjects(subject, current_ROI))
voxels_scores = {}
coefs = []
alphas = []
#Load the data
print('Loading data ...')
data = dt.get_data(subject, current_ROI)

# Loop over blocks
r2_tests = []
r2_trains = []
for block, current_block in tqdm(enumerate(data)): # Loop over train-test splits, leaving-out each block at a time
	csvData = []; all_models = []; alphas_block = [];
	predictors, data, X_test, y_test, groups = dt.split_train_test_data(current_block, block)
	
	# Run PCA on features
	if pref.compute_PCA: predictors, X_test = train.compute_PCA(predictors, data, X_test)
	
	# Iterate over voxels
	if not args.test: nb_voxels = 1 
	else: nb_voxels = data.shape[1]
	model, out = train.train_model_with_nested_cv(predictors, data, subject, groups, nested)
	r2_test = train.evaluate_model(X_test, y_test, model)
	r2_trains.append(out[3])
	r2_tests.append(r2_test)

	if not os.path.exists(os.path.join(settings.path2Output, 'results/models_blocks', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
		os.makedirs(os.path.join(settings.path2Output, 'results/models_blocks', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))
	pickle.dump(model, open(os.path.join(settings.path2Output,'results/models_blocks', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Block_{}.pkl'.format(block +1)), 'wb'))

# Clean R2scores
r2_trains = np.mean(np.vstack(r2_trains), axis=0)
r2_trains[r2_trains < 0], r2_trains[r2_trains > 0.99] = 0, 0 
r2_tests = np.mean(np.vstack(r2_tests), axis=0)
r2_tests[r2_tests < 0], r2_tests[r2_tests > 0.99] = 0, 0
	
# Plotting
if pref.subset == None:
	print('Figure ...')
	plot.glass_brain(plot.mask_inverse(r2_tests, subject, current_ROI), subject, name_ROI, filename)



		# for voxel in range(nb_voxels):
		# #for voxel in range(nb_voxels):
		# 	model, modeln, out = train.train_model_with_nested_cv(predictors, data, subject, voxel, groups)
		# 	r2_test = train.evaluate_model(X_test, y_test[:,voxel], model)
		# 	csvData.append(out + [r2_test])
		# 	all_models.append([model, modeln])
		# 	alphas_block.append(out[1])

		# 	if pref.ridge_nested_crossval: plot.regularization_path(out[2], out[3], subject, voxel, block, name_ROI)
			
		# if block == 0: voxels_scores[voxel] = [r2_test]
		# else: voxels_scores[voxel] = voxels_scores[voxel] + [r2_test]

		

		# if not os.path.exists(os.path.join(settings.path2Output, 'results/models', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
		# 	os.makedirs(os.path.join(settings.path2Output, 'results/models', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))
		# pickle.dump(model, open(os.path.join(settings.path2Output,'results/models', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Block_{}.pkl'.format(block +1)), 'wb'))
		
		# # Generate csv files
		# if pref.csv_block:		
		# 	if not os.path.exists(os.path.join(settings.path2Output, 'results/r_squared', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
		# 		os.makedirs(os.path.join(settings.path2Output, 'results/r_squared', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))	
			
		# 	with open(os.path.join(settings.path2Output, 'results/r_squared', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Block_{}.csv'.format(block +1)), 'w') as csvfile:
		# 		writer = csv.writer(csvfile)
		# 		writer.writerow(['voxel', 'best_alpha', 'r2_train_ncv', 'r2_val', 'r2_train_cv', 'r2_test'])
		# 		writer.writerows(csvData)
		# alphas.append(np.mean(np.array(alphas_block)))


	# r2_tests = np.mean(np.vstack(r2_tests), axis=0)
	# print(r2_tests.shape) 

	
	# alphas = np.mean(np.array(alphas))
	# print(alphas)

	# if not os.path.exists(os.path.join(settings.path2Output, 'results/alpha', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
	# 			os.makedirs(os.path.join(settings.path2Output, 'results/alpha', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))	
	# pickle.dump(alphas, open(os.path.join(settings.path2Output, 'results/alpha', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'alpha_{}.pkl'.format(subject)), 'wb'))
	
	# if not os.path.exists(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
	# 			os.makedirs(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))	
	# pickle.dump(voxels_scores, open(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Voxels_scores_{}.pkl'.format(subject)), 'wb'))
	# voxels_scores = pickle.load(open(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Voxels_scores_{}.pkl'.format(subject)), 'rb'))

	# Generate list of r2_test scores with voxel number as index
	# r2_tests = []
	# for i, voxel in enumerate(voxels_scores):
	# 	r2_tests.append(np.mean(np.array(voxels_scores[voxel])))

