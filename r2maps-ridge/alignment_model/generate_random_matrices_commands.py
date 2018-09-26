import os
for i in range(0, 112):
	os.system('python3 main.py -s 58 -f shuffle_basic_lstm -fo rms* bottom* freq* f0* wordr* lstm_* --shuffle {}'.format(i))

