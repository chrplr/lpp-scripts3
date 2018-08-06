####################### PARAMETERS #######################
"""
ROIs :
   -1. None
	0. IFGorb
	1. IFGtri
	2. TP
	3. TPJ
	4. aSTS
	5. pSTS

Non-LSTM features :
	1601. RMS
	1602. f0
	1603. Word
	1604. Frequency
	1605. BottomUp


About Experiment : 51 subjects, 9 blocks per subject, around 300 scans per subject, 219486 voxels per scan.
"""
##########################################################
import numpy as np

class settings:
	def __init__(self):
		# Paths
		self.rootpath = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/'
		self.path2Code = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Code'
		self.path2Data = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Data'
		self.path2Output = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Output'
		self.path2Figures = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Figures'
		self.path2local = '/home/av256874/Documents'
		self.path2logs = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Code/alignment_model/Logs'

class preferences:
	def __init__(self):

		# Number of subjects
		self.testing_one_subject = True	
		self.testing_ten_subjects = False
		
		# Crossvalidation prefernces
		self.ridge_nested_crossval = True
		
		# Data
		self.generate_data = True
		self.compute_PCA = True
		self.n_components = 4

		#Plotting preferences
		self.save_figures = True
		self.plot_figures = True
		self.test_plot = False
		self.plot = True
		self.name = '300wepca4_5rms'

		# Save data
		self.csv_sub = False
		self.csv_block = True

		# Debug
		self.print_steps = True 



class params:
	def __init__(self):
		pref = preferences()

		# Data
		self.nb_blocks = 9
		self.scans = [282, 298, 340, 303, 265, 343, 325, 292, 368]
		self.nb_features = 1300
		self.features_of_interest = list(range(1301, 1606)) # + list(range(100, 120))))
		self.subset = 20
		
		# Plot
		self.style_plot = 'fivethirtyeight'

		# Crossvalidation
		self.method_nested = 'GridSearchCV'
		self.method = 'Ridge'
		self.defaut_alpha = 15

		# Alpha for nested
		self.n_alphas = 20
		self.alpha_order_min = -4
		self.alpha_order_max = 6
		self.alphas_nested_ridgecv = np.logspace(self.alpha_order_min, self.alpha_order_max, self.n_alphas)
		self.fit_intercept = True
		
		# Subjects 
		if not pref.testing_one_subject and not pref.testing_ten_subjects :
			self.subjects = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 
						87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
		
		if pref.testing_one_subject and not pref.testing_ten_subjects: self.subjects = [57]
		
		if pref.testing_ten_subjects: self.subjects = [59, 61, 62, 63, 64, 65, 66, 67, 68, 69] 


