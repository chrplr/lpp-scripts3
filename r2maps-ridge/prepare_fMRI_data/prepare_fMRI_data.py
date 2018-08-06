import os
import glob
import numpy as np 
import pickle
from tqdm import tqdm

import nilearn
import nibabel as nib
# from nilearn.image import index_img
from nilearn.masking import compute_epi_mask
# from nilearn.masking import apply_mask


nb_blocks = 9
scans = [282, 298, 340, 303, 265, 343, 325, 292, 368]
subjects = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 95, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
#subjects = [70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 
# subjects = [96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
rootdir = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI'
files = [glob.glob(os.path.join(rootdir, "fmri-data/en", "sub-%03d" % s, "func","*.nii")) for s in subjects]

for subject in tqdm(range(len(subjects)), total=len(subjects), unit='subject'):
	for file in tqdm(range(nb_blocks), total=nb_blocks, unit='block'):
		run = nib.load(files[subject][file])

		with open('../../Data/en/fmri_data/masks/sub_{0}/{1}.pkl'.format(subjects[subject], files[subject][file][-34:-4]), 'wb') as fout:
			#pickle.dump([run.get_data(), compute_epi_mask(run)], fout)
			pickle.dump(compute_epi_mask(run), fout)





