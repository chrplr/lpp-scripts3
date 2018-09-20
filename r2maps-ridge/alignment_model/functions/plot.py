import os
import glob
import pickle
import numpy as np

from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import math_img, mean_img
from nilearn.plotting import plot_glass_brain

import matplotlib.pyplot as plt

from functions import settings
paths = settings.paths()


def glass_brain(r2_voxels, subject, current_ROI, ROI_name, name):
	"""
	Input : Masked results of r2score
	Take masked data and project it again in a 3D space
	Ouput : 3D glassbrain of r2score 
	"""

	# Get one mask and fit it to the corresponding ROI
	if current_ROI != -1 and current_ROI <= 5:
		masks_ROIs_filenames = sorted(glob.glob(os.path.join(paths.path2Data, 
												"en/ROIs_masks/",
												"*.nii")))
		ROI_mask = masks_ROIs_filenames[current_ROI]
		ROI_mask = NiftiMasker(ROI_mask, detrend=True, standardize=True)
		ROI_mask.fit()
		unmasked_data = ROI_mask.inverse_transform(r2_voxels)

	# Get masks and fit a global mask
	else:
		masks = []
		masks_filenames = sorted(glob.glob(os.path.join(paths.path2Data, 
													"en/fmri_data/masks",
													"sub_{}".format(subject),  
													"resample*.pkl")))
		for file in masks_filenames:
			with open(file, 'rb') as f:
				mask = pickle.load(f)
				masks.append(mask)

		global_mask = math_img('img>0.5', img=mean_img(masks))
		masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
		masker.fit()
		unmasked_data = masker.inverse_transform(r2_voxels)

	display = plot_glass_brain(unmasked_data, display_mode='lzry', threshold='auto', colorbar=True, title='Sub_{}'.format(subject))
	if not os.path.exists(os.path.join(paths.path2Figures, 'glass_brain', 'Sub_{}'.format(subject), ROI_name)):
		os.makedirs(os.path.join(paths.path2Figures, 'glass_brain', 'Sub_{}'.format(subject), ROI_name))
	
	display.savefig(os.path.join(paths.path2Figures, 'glass_brain', 'Sub_{}'.format(subject), ROI_name, 'R_squared_test_{}.png'.format(name)))
	print('Figure Path : ', os.path.join(paths.path2Figures, 'glass_brain', 'Sub_{}'.format(subject), ROI_name, 'R_squared_test_{}.png'.format(name)))
	display.close()


def regularization_path(r2_train_average, r2_val_average, subject, block, ROI_name, name):
	"""
	Input : Output of the training
	Plot figures of R-squared error of the tran and validation test depending on alpha 
	Output : Figures
	"""
	alphas = np.logspace(alphas[0], alphas[1], alphas[2])

	if not os.path.exists(os.path.join(paths.path2Figures, 'regularization_path', 'Sub_{}'.format(subject), ROI_name, name)):
		os.makedirs(os.path.join(paths.path2Figures, 'regularization_path', 'Sub_{}'.format(subject), ROI_name, name))
	
	for voxel in range(r2_train_average.shape[1]):
		r2_train = r2_train_average[:, voxel]
		r2_val = r2_val_average[:, voxel]

		fig, ax1 = plt.subplots()
		
		# Set parameters for figure 1 : R-squared train data depending of alpha
		plt.title('Subject_{}'.format(subject))
		ax1.set_xscale('log')
		ax1.set_xlabel("Alpha", size=18)
		ax1.set_ylabel("r2 training set", size=18)
		
		# Set parameters for figure 2 : R-squared validation data depending of alpha
		ax2 = ax1.twinx()
		scores = r2_val
		ax2.plot(alphas, scores, 'r.', label='R-squared test set')
		ax2.set_ylabel('R-squared validation', color='r', size=18)
		ax2.set_ylim(-1,1)
		ax2.tick_params('y', colors='r')
		scores_train = r2_train
		ax2.plot(alphas, scores_train, 'g.', label='R-squared training set')
		plt.axis('tight')
		plt.legend(loc=0)

		# Save the figure
		plt.savefig(os.path.join(paths.path2Figures, 'regularization_path', 'Sub_{}'.format(subject), ROI_name, name, '{0}_voxel_{1}_score'.format(block +1 , voxel +1)))
		plt.close()


def plot_pca(nb_features, ratio, subject, ROI_name, threshold=0.75):
	x, y = list(range(nb_features)), ratio
	plt.plot(x, y)
	i, sum_ = 0, 0
	while sum_ < threshold:
		sum_ = sum_ + y[i]
		i = i + 1
	plt.plot([i, i], [0, 0.1], 'r--', lw=2)
	print("Number of PCA components to reach threshold : ", i)
	if not os.path.exists(os.path.join(paths.path2Figures, 'PCA', 'Sub_{}'.format(subject))):
		os.makedirs(os.path.join(paths.path2Figures, 'PCA', 'Sub_{}'.format(subject)))
	plt.savefig(os.path.join(paths.path2Figures, 'PCA', 'Sub_{}'.format(subject), '{}_pca.png'.format(ROI_name)))
