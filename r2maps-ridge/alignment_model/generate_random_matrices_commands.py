import os
for i in range(0, 112):
	os.system('python3 main.py -s 59 -f shuffle_basic_features -fo rms* bottom* freq* f0* wordr* --shuffle {} &'.format(i))
	os.system('python3 main.py -s 59 -f shuffle_basic_lstm -fo rms* bottom* freq* f0* wordr* lstm_* --shuffle {} &'.format(i))
	os.system('python3 main.py -s 59 -f shuffle_basic_WE_3000 -fo rms* bottom* freq* f0* wordr* word_* --shuffle {}'.format(i))

