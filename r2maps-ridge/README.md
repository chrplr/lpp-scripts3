# 1. Generating LSTM activations
-------------------------------------------
Run pre-trained LSTm model on Le Petit Prince text (.txt files containing tokens (word an punctuation))
Yield activations (one vector of 1300 dimensions per token)  

      cd prepare_LSTM_features/extract_activations/

# 2 . Generating design matrices and perform Ridge crossvalidation 
-------------------------------------------

  cd alignment_model/functions/
  edit settings_params_perferences.py
  python main.py 

This performs:

* Merge LSTM's activations with offsets
* Run event2reg to perform HRF convolution
* Concatenation of regressor's files to create design matrices
* Ridge corss validation 

# 3. Output figures
	
	cd alignment_model/figure_plot.py

