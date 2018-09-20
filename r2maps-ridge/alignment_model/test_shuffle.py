import pickle
import numpy as np
a = np.random.randn(219486,)
with open('functions/all_shuffled_matrices.pkl', 'rb') as f:
	shuffled = np.array(pickle.load(f))
	print(shuffled.shape[1])
	print(shuffled[:, 0])
	p_values_mask = []
	for i in range(shuffled.shape[1]):
		p_value_bool = np.greater((sum(np.greater(shuffled[:,i], a[i])) + 1) / (shuffled.shape[0] + 1), 0.05)
		p_values_mask.append(p_value_bool)
	print(sum(p_values_mask))
	

