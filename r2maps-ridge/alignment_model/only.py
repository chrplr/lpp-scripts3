import pickle
from functions import plot
with open('save_r2test/only_lstm_57.pkl', 'rb') as f:
	r2_test = pickle.load(f)  
print(r2_test)	
plot.glass_brain(r2_test, 57, -1, 'All', 'only_lstm')
