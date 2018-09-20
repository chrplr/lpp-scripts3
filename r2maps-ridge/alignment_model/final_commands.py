import os 

subjects = [57, 58, 59, 61, 62]
for subject in subjects:
	os.system('python3 -W ignore main.py -s {0} -r -1 -n -f basic_features -fo rms* bottom* freq* f0* wordr* &'.format(subject)) 
	os.system('python3 -W ignore main.py -s {0} -r -1 -n -f basic_WE_300 -fo rms* bottom* freq* f0*  wordr* word_* &'.format(subject))
	os.system('python3 -W ignore main.py -s {0} -r -1 -n -f basic_lstm -fo rms* bottom* freq* f0* wordr* lstm_*'.format(subject))
