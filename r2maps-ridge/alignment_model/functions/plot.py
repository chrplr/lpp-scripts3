import os
import glob
import pickle
import numpy as np

from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import math_img, mean_img
from nilearn.plotting import plot_glass_brain

import matplotlib.pyplot as plt

from functions import settings_params_preferences as spp
settings = spp.settings()
params = spp.params()
pref = spp.preferences()


def mask_inverse(r2_voxels, subject, current_ROI):
	"""
	Input : Masked results of r2score
	Take masked data and project it again in a 3D space
	Ouput : 3D data of r2score 
	"""

	# Get one mask and fit it to the corresponding ROI
	if current_ROI != -1 and current_ROI <= 5:
		masks_ROIs_filenames = sorted(glob.glob(os.path.join(settings.path2Data, 
												"en/ROIs_masks/",
												"*.nii")))
		ROI_mask = masks_ROIs_filenames[current_ROI]
		ROI_mask = NiftiMasker(ROI_mask, detrend=True, standardize=True)
		ROI_mask.fit()
		unmasked_data = ROI_mask.inverse_transform(r2_voxels)

	# Get masks and fit a global mask
	else:
		masks = []
		masks_filenames = sorted(glob.glob(os.path.join(settings.rootpath, 
													"Data/en/fmri_data/masks",
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

	return unmasked_data

def glass_brain(img, subject, ROI_name):
	"""
	Imput : 3D data of r2 score
	Plot r2 score on a glass brain
	Output : Figure
	"""
	display = plot_glass_brain(img, display_mode='lzry', threshold='auto', colorbar=True, title='Sub_{}'.format(subject))
	if not os.path.exists(os.path.join(settings.path2Figures, 'results/glass_brain', 'Sub_{}'.format(subject), ROI_name)):
		os.makedirs(os.path.join(settings.path2Figures, 'results/glass_brain', 'Sub_{}'.format(subject), ROI_name))
	display.savefig(os.path.join(settings.path2Figures, 'results/glass_brain', 'Sub_{}'.format(subject), ROI_name, 'R_squared_test_{}.png'.format(pref.name)))
	display.close()


def regularization_path(r2_train, r2_test, subject, voxel, block, ROI_name):
	"""
	Input : Output of the training
	Plot figures of R-squared error of the tran and validation test depending on alpha 
	Output : Figures
	"""
	#Declare figures
	fig, ax1 = plt.subplots()
	
	# Set parameters for figure 1 : R-squared train data depending of alpha
	plt.title('Subject_{}'.format(subject))
	ax1.set_xscale('log')
	ax1.set_xlabel("Alpha", size=18)
	ax1.set_ylabel("r2 training set", size=18)
	
	# Set parameters for figure 2 : R-squared validation data depending of alpha
	ax2 = ax1.twinx()
	scores = r2_test
	ax2.plot(params.alphas_nested_ridgecv, scores, 'r.', label='R-squared test set')
	ax2.set_ylabel('R-squared validation', color='r', size=18)
	ax2.set_ylim(-1,1)
	ax2.tick_params('y', colors='r')
	scores_train = r2_train
	ax2.plot(params.alphas_nested_ridgecv, scores_train, 'g.', label='R-squared training set')
	plt.axis('tight')
	plt.legend(loc=0)

	# Save the figure
	if pref.save_figures: 
		if not os.path.exists(os.path.join(settings.path2Figures, 'crossval_ridge', 'Sub_{}'.format(subject), ROI_name)):
			os.makedirs(os.path.join(settings.path2Figures, 'crossval_ridge', 'Sub_{}'.format(subject), ROI_name))
		plt.savefig(os.path.join(settings.path2Figures, 'crossval_ridge', 'Sub_{}'.format(subject), ROI_name, '{0}_voxel_{1}_score'.format(block +1 , voxel)))
	plt.close()


def plot_pca(ratio):
	x = [i for i in range(1, 1301)]
	print(x)
	print(ratio)
	y = ratio
	plt.plot(x, y)
	subject = 57
	i = 0
	sum_ = 0
	while sum_ < 0.8:
		sum_ = sum_ + y[i]
		i = i + 1
	print(i)


	if pref.save_figures: 
		if not os.path.exists(os.path.join(settings.path2Figures, 'pca', 'Sub_{}'.format(subject))):
			os.makedirs(os.path.join(settings.path2Figures, 'pca', 'Sub_{}'.format(subject)))
		plt.savefig(os.path.join(settings.path2Figures, 'pca', 'Sub_{}'.format(subject), 'pca.png'))