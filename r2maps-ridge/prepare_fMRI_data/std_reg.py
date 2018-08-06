import pickle
import csv
from tqdm import tqdm
import numpy as np 

onsets = []
activation_words = []
nb_blocks = 9
csvData = []

for block in tqdm(range(9)):
	for feature in tqdm(range(1300)):
		with open('../../Output/regressors/en/Block{0}/{1}_TimeFeat_{2}_reg.csv'.format(block + 1, block + 1, feature + 1), 'r') as readFile:
			reader = csv.reader(readFile)
			lines = list(reader)
			colonne = [float(i[0]) for i in lines]
			np.asarray(colonne)
			colonne = (colonne - np.mean(colonne))/np.std(colonne)
			colonne = colonne.tolist()
			for i in range(len(colonne)):
				csvData.append([colonne[i]])

		with open('../../Output/regressors_std/en/Block{0}/{1}_TimeFeat_{2}_reg.csv'.format(block + 1, block + 1, feature + 1), 'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerows(csvData)

