
#
# Note: This Makefile is not clever, i.e. it  does not know about dependencies
# Time-stamp: <2018-04-08 14:43:15 cp983411>

SHELL := /bin/bash

export PATH := bin:$(PATH)
export PYTHONPATH := bin:$(PYTHONPATH)

# DO NOT EDIT THIS PART (unless you know what you are doing)
# set the environment variables in the shell before calling `make`
export LINGUA ?= en
export REGS ?=  bottomup freq rms wordrate
export MODEL ?= rms-wordrate-freq-bottomup
export MODEL_DIR ?= models/$(LINGUA)/$(MODEL)

export SUBJECTS_FMRI_DATA ?= $(ROOT_DIR)/fmri-data/$(LINGUA)
export ONSETS_DIR ?= inputs/onsets/$(LINGUA)
export REGS_DIR ?= outputs/regressors/$(LINGUA)
export DESIGN_MATRICES_DIR = outputs/design-matrices/$(LINGUA)/$(MODEL)
export FIRSTLEVEL_RESULTS ?= outputs/results-indiv/$(LINGUA)/$(MODEL)
export GROUP_RESULTS ?= outputs/results-group/$(LINGUA)/$(MODEL)
export ROI_RESULTS ?= outputs/results-group/$(LINGUA)/$(MODEL)-roi

regressors:
	mkdir -p $(REGS_DIR)
	python generate-regressors.py --no-overwrite --input-dir $(ONSETS_DIR) --output-dir $(REGS_DIR) $(REGS)

design-matrices:
	mkdir -p $(DESIGN_MATRICES_DIR); \
	python merge-regressors.py -i $(REGS_DIR) -o $(DESIGN_MATRICES_DIR) $(REGS)
	if [ -f $(MODEL_DIR)/orthonormalize.py ]; then \
		echo 'Orthogonalizing...'; \
		python $(MODEL_DIR)/orthonormalize.py \
			--design_matrices=$(DESIGN_MATRICES_DIR) \
			--output_dir=$(DESIGN_MATRICES_DIR); \
	fi
	for f in $(DESIGN_MATRICES_DIR)/*.csv; do python check-design-matrices.py $$f >$${f%.csv}_diagnostics.txt; done

first-level:
	mkdir -p $(FIRSTLEVEL_RESULTS); \
	python $(MODEL_DIR)/firstlevel.py \
		--subject_fmri_data=$(SUBJECTS_FMRI_DATA) \
		--design_matrices=$(DESIGN_MATRICES_DIR) \
		--output_dir=$(FIRSTLEVEL_RESULTS)

second-level:
	mkdir -p $(GROUP_RESULTS); \
	python $(MODEL_DIR)/group.py \
		--data_dir=${FIRSTLEVEL_RESULTS} \
		--output_dir=$(GROUP_RESULTS) 

roi-analyses:
	python lpp-rois.py --data_dir=${FIRSTLEVEL_RESULTS} --output=$(MODEL)-rois.csv
