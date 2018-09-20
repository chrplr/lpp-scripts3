################################################################################################
# Generate commands to reproduce figures
################################################################################################
import os
import argparse

parser = argparse.ArgumentParser(description='Run or not')
parser.add_argument('--run', '-r', action='store_true',
					help='Run all the commands')
args = parser.parse_args()


# Set experiments, subjects and ROIs
name_experiments = ['rms', 'wordrate', 'freq', 'f0', 'bottomup', 'basic_features', 'basic_WE_300', 'basic_75pca_lstm', 'basic_lstm']
features_experiments = ['-fo rms* wordrate* freq* f0* bottomup*',
			'-fo rms* wordrate* freq* f0* bottomup* word_*',
			'-fo rms* wordrate* freq* f0* bottomup* lstm_*']
subjects = [57, 58, 59, 61, 62]
ROIs = [-1]

# Print all the commands
for n_experiment, experiment in enumerate([name_experiments[5], name_experiments[6], name_experiments[8]]):			# rms, wordrate, frequency, f0, bottomup, basic_features, basic_lstm, basic_WE_300, basic_75_pca_lstm
	for subject in subjects: 
		for ROI in ROIs:
			cmd = 'python3 -W ignore main.py -s {0} -r {1} -n -f {2} {3} &'.format(subject, ROI, experiment, features_experiments[n_experiment])
			if args.run:
				os.system(cmd)
			else:
				print(cmd)


