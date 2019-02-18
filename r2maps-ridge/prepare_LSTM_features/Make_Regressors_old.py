# Make regressors files by convolute activations with Hemodynamic Response Function (HRF)

import os 
import sys


nb_scans = [282, 298, 340, 303, 265, 343, 325, 292, 368]


if len(sys.argv) > 1:
	print ('Block ' + sys.argv[1])
	block = int(sys.argv[1])
else:
	block = 1 

features = 1300
for file in range(features):
	filename = '{0}_TimeFeat_{1}'.format(block, file + 1)
	filepath = 'Block{}/'.format(block) + filename +'.csv'
	command = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/scripts-python/bin/onsets2reg.old -t 2.0 -n {} -i '.format(nb_scans[block - 1]) + filepath + ' -o ../../Regressors/en/' + filename + '_reg.csv'
	print(command)
	os.system(command)

