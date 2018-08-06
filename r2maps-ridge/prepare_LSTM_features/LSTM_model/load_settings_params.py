import os.path as op
import numpy as np

class settings:
    def __init__(self):

        # Regression
        self.method = 'Ridge'
        self.y_label = 'all'  # Which label to regress from the meta text data
        self.h_or_c = 1  # zero or one. 0: hidden 1: cell
        self.which_layer = 1  # 0: both, 1: first, 2: second

        # Paths
        self.path2code = '/neurospin/unicog/protocols/intracranial/FAIRNS/sentence-processing-MEG-LSTM/Code/MEG'
        #self.path2code = '/home/yl254115/Projects/FAIRNS/sentence-processing-MEG-LSTM/Code/LSTM/model-analysis'
        self.path2LSTMdata = op.join('..', '..', '..', 'Data', 'LSTM')
        self.path2figures = op.join('..', '..', '..', 'Figures')
        self.path2output = op.join('..', '..', '..', 'Output')

        # Files
       #self.LSTM_file_name = 'vectors-LSTM1000-0.npy'
        self.LSTM_file_name = 'vectors-LSTM500_2-0.npz'
        self.bnc_data = 'bnc_0313_filtered.pkl'
        self.LSTM_pretrained_model = 'hidden650_batch128_dropout0.2_lr20.0.cpu.pt'
        self.LSTM_pretested_file_name = 'LSTM_activations_pretested_on_sentences_' + self.LSTM_pretrained_model + '_h_or_c_' + str(self.h_or_c)  + '.pkl'
        self.vocabulary_file = 'english_vocab.txt'
        self.eos_separator = "<eos>"
        # Flags



class params:
    def __init__(self):
        #
        self.seed_split = 3 # random seed for split
        self.CV_fold = 5  # 5-fold

        # Hyper-parameters regression:
        self.n_alphas = 50 # regularization size
        self.alpha_order_min = -6 # 10^(order) range for regularization size search
        self.alpha_order_max = 3 # same
        self.alphas = np.logspace(self.alpha_order_min, self.alpha_order_max, self.n_alphas)
        self.eps = 1e-3 # see Scikit-learn
        self.l1_ratio = 0.8 # For Elastic-Net

class preferences:
    def __init__(self):
        self.run_Ridge = True
        self.run_LASSO = False
        self.run_ElasticNet = False
        self.load_pretested_LSTM = False
