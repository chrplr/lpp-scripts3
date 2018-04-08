scripts for the analysis of the fMRI data of "The Little Prince" project (Hale, Pallier)
----------------------------------------------------------------------------------------

Christophe@pallier.org

Before anything, set the ROOT_DIR of the project, for example:

    . setroot-neurospin

To perform an analysis, first select a model in `models` to set several environment variables:

    . setmodel models/en/chrmodels/en/rms-wordrate-freq-bottomup/

Then, you can execute the analysis step by step, following the stages in the `Makefile`:
	
	make regressors    # check outputs/regressors
	make design-matrices   # check outputs/design-matrices
	make first-level   # check output/
	make second-level
	make roi-analyses
 

To create a new model, you need to:

1. create a subdirectory inside the `models` directory

2. create a `setenv` file exporting the environment variables specifing the model name, the list of regressors, etc. (see modesl/en/christophe-bottomup/setenv fro an example)

3. In your model's directory, create `firstlevel.py` and optionaly, `orthonormalize.py` which will be executed by `make first-level`. Create also `group.py` for the second level.

4. If your model includes variables that have not yet been used in previous models, you need to add to the folder `inputs/onsets`  one comma separated (.csv) file per variable and per run --- the filename pattern being `X_VARNAME.csv` where X is the run number [1-9]. Each file must contain two columns named 'onset' and 'amplitude' (onsets is given in seconds). 

Requirements:

- Python: pandas, nistats, nibabel, nilearn
- R: car, rmarkdown (only for make check-design-matrices)
