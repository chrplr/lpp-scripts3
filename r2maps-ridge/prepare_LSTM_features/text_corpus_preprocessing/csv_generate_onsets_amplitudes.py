import pickle
import csv
from tqdm import tqdm
import numpy as np 

onsets = []
activation_words = []
nb_blocks = 9
csvData = []

for block in tqdm(range(1,nb_blocks +1), total=nb_blocks, unit='block'):
	with open('../../../Data/en/word_onset_times/eng_{}.csv'.format(block), 'r') as readFile:
		reader = csv.reader(readFile)
		lines = list(reader)
		for line in lines:
			onset = line[3] # 3 for eng, 2 for fr
			onsets.append(onset)
	readFile.close()

	with open("../../../Output/patterns/en/Block{}_activations.pkl".format(block), 'rb') as f:
		x = pickle.load(f)

		features = len(x['hidden'][0])
		words = len(x['hidden'][0][0])

	for feature in tqdm(range(features), total=features, unit='feature'):	
		for activation_word in range(words):
			activation_word = x['hidden'][0][feature][activation_word]
			activation_words.append(activation_word)

	# Use only for standardization	
		np.asarray(activation_words)
		activation_words = (activation_words - np.mean(activation_words))/np.std(activation_words)
		activation_words.tolist()
	###############
		
		for time in range(len(onsets)):
			csvData.append([onsets[time], activation_words[time]])

		with open('../../../Data/en/features_std/Block{0}/{1}_TimeFeat_{2}.csv'.format(block, block, feature + 1), 'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['onset', 'amplitude'])
			writer.writerows(csvData)

		csvData = []
		activation_words = []
	
	onsets = []