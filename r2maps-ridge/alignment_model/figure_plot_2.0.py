import matplotlib.pyplot as plt 
import pickle
from tqdm import tqdm
import numpy as np
from functions import plot
import argparse

parser = argparse.ArgumentParser(description='Parameters for computing')
parser.add_argument('--p_value', '-p', type=float, default=0.05,
					help='Choose a p-value')
args = parser.parse_args()


plt.style.use('fivethirtyeight')

# Retrieve data

def retrieve_data(name):
	retrieved_data = {}
	with open('shuffle/all_shuffle_{}.pkl'.format(name), 'rb') as f:
		retrieved_data[name] = np.amax(np.vstack(pickle.load(f)), axis=1)

retrieve_data('basic_features')
retrieve_data('basic_WE_300')
retrieve_data('basic_LSTM')

def clean_data(name):
	retrieve_data[name][retrieve_data[name] > 0.99], retrieve_data[name][retrieve_data[name] < 0.0] = 0, 0

clean_data('basic_features')
clean_data('basic_WE_300')
clean_data('basic_LSTM')

# Clean by significance
subjects = [57]

for subject in subjects:
	with open('save_r2test/basic_features_{}.pkl'.format(subject), 'rb') as f:
		r2_test_basic_features = pickle.load(f)

	with open('save_r2test/basic_WE_300_{}.pkl'.format(subject), 'rb') as g:
		r2_test_basic_WE_300 = pickle.load(g)

	with open('save_r2test/basic_lstm_{}.pkl'.format(subject), 'rb') as h:
	 	r2_test_basic_LSTM = pickle.load(h)


	greater_BF = [(np.sum(np.greater(random_r2tests_basic_features, r2_test_basic_features[voxel])) + 1) / (len(random_r2tests_basic_features + 1)) for voxel in tqdm(range(len(r2_test_basic_features)))]
	mask_BF = np.less_equal(greater_BF, args.p_value)
	r2_test_basic_features = np.multiply(r2_test_basic_features, mask_BF)

	greater_WE = [(np.sum(np.greater(random_r2tests_basic_WE_300, r2_test_basic_WE_300[voxel])) + 1) / (len(random_r2tests_basic_WE_300) + 1) for voxel in tqdm(range(len(r2_test_basic_WE_300)))]
	mask_WE = np.less_equal(greater_WE, args.p_value)
	r2_test_basic_WE_300 = np.multiply(r2_test_basic_WE_300, mask_WE)

	greater_LSTM = [(np.sum(np.greater(random_r2tests_basic_LSTM, r2_test_basic_LSTM[voxel])) + 1) / (len(random_r2tests_basic_LSTM) + 1) for voxel in tqdm(range(len(r2_test_basic_LSTM)))]
	mask_LSTM = np.less_equal(greater_LSTM, args.p_value)
	r2_test_basic_LSTM = np.multiply(r2_test_basic_LSTM, mask_LSTM)

	plot.glass_brain(r2_test_basic_features, subject, -1, 'All', 'basic_features_significant_{}'.format(args.p_value))
	plot.glass_brain(r2_test_basic_WE_300, subject, -1, 'All', 'basic_WE_300_significant_{}'.format(args.p_value))
	plot.glass_brain(r2_test_basic_LSTM, subject, -1, 'All', 'basic_LSTM_significant_{}'.format(args.p_value))

	scatter_1 = [i for i in r2_test_basic_features if i != 0]
	scatter_2 = [r2_test_basic_LSTM[i] for i, h in enumerate(r2_test_basic_features) if h != 0]
	scatter_3 = [i for i in r2_test_basic_WE_300 if i != 0]
	scatter_4 = [i for i in r2_test_basic_WE_300 if i != 0]
	scatter_5 = [r2_test_basic_WE_300[i] for i, h in enumerate(r2_test_basic_features) if h != 0]
	scatter_6 = [r2_test_basic_LSTM[i] for i, h in enumerate(r2_test_basic_WE_300) if h != 0]
	
	ratio = np.array(scatter_2)  > np.array(scatter_1)
	print(np.sum(ratio)/len(scatter_1))
	
	
	r2_test_basic_features = [i for i in r2_test_basic_features if i != 0]
	r2_test_basic_WE_300 = [j for j in r2_test_basic_WE_300 if j != 0]
	r2_test_basic_LSTM = [h for h in r2_test_basic_LSTM if h != 0]
	
	
	print('Number of significant voxels for basic_features : ', len(r2_test_basic_features))
	print('Number of significant voxels for basic_WE_300 : ', len(r2_test_basic_WE_300))
	print('Number of significant voxels for basic_LSTM : ', len(r2_test_basic_LSTM))

	#plt.hist(r2_test_basic_features, bins=100, alpha=0.5, label='basic_features')
	#plt.hist(r2_test_basic_WE_300, bins=100, alpha=0.5, label='basic_WE_300')
	#plt.hist(r2_test_basic_LSTM, bins=100, alpha=0.5, label='basic_LSTM')
	plt.hist([r2_test_basic_features, r2_test_basic_WE_300, r2_test_basic_LSTM], bins=100, label=['basic_features', 'basic_WE_300', 'basic_LSTM'])
	plt.tight_layout()
	plt.legend(loc='upper right')
	plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge/figures/histograms/Sub_{}.png'.format(subject))
	plt.close()

	plt.scatter(scatter_1, scatter_2, marker='o', s=0.2)
	plt.plot([0, 0.25], [0, 0.25], 'r--', linewidth=0.5)
	plt.xlim(0, 0.25)
	plt.ylim(0, 0.25)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.xlabel('Basic features')
	plt.ylabel('Basic + LSTM')
	plt.title('Best model')
	plt.tight_layout()
	plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge/figures/scatter_plot/Sub_{}_BF_LSTM.png'.format(subject))
	plt.close()
	
	plt.scatter(scatter_1, scatter_5, marker='o', s=0.2)
	plt.plot([0, 0.25], [0, 0.25], 'r--', linewidth=0.5)
	plt.xlim(0, 0.25)
	plt.ylim(0, 0.25)
	plt.xlabel('Basic features')
	plt.ylabel('Basic + WE')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.title('Best model')
	plt.tight_layout()
	plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge/figures/scatter_plot/Sub_{}_BF_WE.png'.format(subject))
	plt.close()

	plt.scatter(scatter_3, scatter_6, marker='o', s=0.2)
	plt.plot([0, 0.25], [0, 0.25], 'r--', linewidth=0.5)
	plt.xlim(0, 0.25)
	plt.ylim(0, 0.25)
	plt.xlabel('Basic + WE')
	plt.ylabel('Basic + LSTM')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.title('Best model')
	plt.tight_layout()
	plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge/figures/scatter_plot/Sub_{}_WE_LSTM.png'.format(subject))
	plt.close()
