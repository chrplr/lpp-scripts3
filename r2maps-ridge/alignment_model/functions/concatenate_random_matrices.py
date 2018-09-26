import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Concatenate all random r_squared simulations')

parser.add_argument('--nb_runs', '-nr', type=int, default=112,
					help='Number of runs to concatenate')
parser.add_argument('--nb_blocs', '-nb', type=int, default=9,
					help='Number of blocks')
parser.add_argument('--filename', '-f', default='no_filename', 
					help='Name of the file')
args = parser.parse_args()


def get_shuffled_matrices(nb_runs, nb_blocks, name):
	shuffle_matrices = []
	for run in range(nb_runs):
		with open('../shuffle/shuffle{}.pkl'.format(run), 'rb') as f:
			shuffle_matrices.append(pickle.load(f))	
	shuffle_matrices = [shuffle_matrices[i][j] for i in range(nb_runs) for j in range(nb_blocks)]
	with open('../shuffle/all_{}.pkl'.format(name), 'wb') as f:
		pickle.dump(shuffle_matrices, f)

get_shuffled_matrices(args.nb_runs, args.nb_blocks, args.filename)


