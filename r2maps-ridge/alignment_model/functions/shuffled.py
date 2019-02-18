import pickle
import numpy as np

def get_shuffled_matrices(nb_runs, nb_blocks, name, subject):
	shuffle_matrices = []
	for run in range(nb_runs):
		with open('../shuffle/shuffle_{}_{}_{}.pkl'.format(run, name, subject), 'rb') as f:
			shuffle_matrices.append(pickle.load(f))	
	shuffle_matrices = [shuffle_matrices[i][j] for i in range(nb_runs) for j in range(nb_blocks)]
	with open('../shuffle/all_{}_{}.pkl'.format(name, subject), 'wb') as f:
		pickle.dump(shuffle_matrices, f)

get_shuffled_matrices(112, 9, 'shuffle_basic_lstm', 58)
get_shuffled_matrices(112, 9, 'shuffle_basic_lstm', 59)
get_shuffled_matrices(112, 9, 'shuffle_basic_WE_3000', 58)
get_shuffled_matrices(112, 9, 'shuffle_basic_WE_3000', 59)



