import os
import csv
import sys
import pickle 

import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm

from functions import data as dt
from functions import train
from functions import plot
from functions import settings_params_preferences as spp

import matplotlib.pyplot as plt

print('Loading settings ...')
settings = spp.settings()
print('Loading parameters ...')
params = spp.params()
print('Loading preferences ...')
pref = spp.preferences()

print('Subjects : ', params.subjects)

plt.switch_backend('Agg')

# Get ROI from argument
current_ROI = int(sys.argv[1])
ROI = ['IFGorb', 'IFGtri', 'TP', 'TPJ', 'aSTS', 'pSTS']
name_ROI = ROI[current_ROI]
if current_ROI == -1: name_ROI = 'All'

# Compute crossvalidation for each subject
for subject in params.subjects:
	if pref.print_steps: print('Generating data ...')
	if pref.generate_data: dt.generate_data_all_subjects(subject, current_ROI)
	voxels_scores = {}
	alphas = []
	#Load the data
	if pref.print_steps: print('Loading data ...')
	data = dt.get_data(subject, current_ROI)
	
	# Loop over blocks
	for block, current_block in enumerate(data): # Loop over train-test splits, leaving-out each block at a time
		csvData = []; all_models = []; alphas_block = [];
		predictors, data, X_test, y_test, groups = dt.split_train_test_data(current_block, block)
		
		# Run PCA on features
		if pref.compute_PCA: predictors, X_test = train.compute_PCA(predictors, data, X_test)
		
		# Iterate over voxels
		if params.subset != None: 
			nb_voxels = params.subset
		else:
			nb_voxels = data.shape[1]

		for voxel in range(nb_voxels):
		#for voxel in range(nb_voxels):
			model, modeln, out = train.train_model_with_nested_cv(predictors, data, subject, voxel, groups)
			r2_test = train.evaluate_model(X_test, y_test[:,voxel], model)
			csvData.append(out + [r2_test])
			all_models.append([model, modeln])
			alphas_block.append(out[1])

			if pref.ridge_nested_crossval: plot.regularization_path(out[2], out[3], subject, voxel, block, name_ROI)
			
			if block == 0: voxels_scores[voxel] = [r2_test]
			else: voxels_scores[voxel] = voxels_scores[voxel] + [r2_test]

		if not os.path.exists(os.path.join(settings.path2Output, 'results/models', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
			os.makedirs(os.path.join(settings.path2Output, 'results/models', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))
		pickle.dump(all_models, open(os.path.join(settings.path2Output,'results/models', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Block_{}.pkl'.format(block +1)), 'wb'))
		
		# Generate csv files
		if pref.csv_block:		
			if not os.path.exists(os.path.join(settings.path2Output, 'results/r_squared', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
				os.makedirs(os.path.join(settings.path2Output, 'results/r_squared', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))	
			
			with open(os.path.join(settings.path2Output, 'results/r_squared', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Block_{}.csv'.format(block +1)), 'w') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(['voxel', 'best_alpha', 'r2_train_ncv', 'r2_val', 'r2_train_cv', 'r2_test'])
				writer.writerows(csvData)
		alphas.append(np.mean(np.array(alphas_block)))
	
	alphas = np.mean(np.array(alphas))
	print(alphas)

	if not os.path.exists(os.path.join(settings.path2Output, 'results/alpha', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
				os.makedirs(os.path.join(settings.path2Output, 'results/alpha', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))	
	pickle.dump(alphas, open(os.path.join(settings.path2Output, 'results/alpha', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'alpha_{}.pkl'.format(subject)), 'wb'))
	
	if not os.path.exists(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI))):
				os.makedirs(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI)))	
	pickle.dump(voxels_scores, open(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Voxels_scores_{}.pkl'.format(subject)), 'wb'))
	voxels_scores = pickle.load(open(os.path.join(settings.path2Output, 'results/scores', 'Sub_{}'.format(subject), '{}'.format(name_ROI), 'Voxels_scores_{}.pkl'.format(subject)), 'rb'))

	# Generate list of r2_test scores with voxel number as index
	r2_tests = []
	for i, voxel in enumerate(voxels_scores):
		r2_tests.append(np.mean(np.array(voxels_scores[voxel])))

	# Clean R2scores
	print(len(r2_tests))
	r2_tests = [r2_tests[i] if r2_tests[i] <= 0.99 and r2_tests[i] > 0 else 0 for i in range(len(r2_tests))]

	r2_tests = np.array(r2_tests)
	print(len(r2_tests))
	# Plotting
	if pref.plot and params.subset == None:
		plot.glass_brain(plot.mask_inverse(r2_tests, subject, current_ROI), subject, name_ROI)

