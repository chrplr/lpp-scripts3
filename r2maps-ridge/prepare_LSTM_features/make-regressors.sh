#! /bin/bash
# Time-stamp: <2019-02-19 13:56:14 christophe@pallier.org>

# Create regressors files by convoluting the activations with the Hemodynamic Response Function (HRF)

EN_NSCANS=(0 282 298 340 303 265 343 325 292 368)  # '0' inserted at the beginning to start indexing at 1

SCRIPTSDIR=/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/

INPUTDIR=/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Data/en/features_std_activation/

# because generate-regressors does not currently accept wildcards in feature names, we have to list all the features:
list_features() {
        OWD=$PWD
        cd $INPUTDIR/Block1
        for f in *.csv;
        do
        a=${f%.csv}
        echo ${a/1_}
        done
        cd $OWD
}

FEATURES=$( list_features )

OUTPUTDIR=/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/regressors/en


python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block1/ --blocks 1 --nscans ${EN_NSCANS[1]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block2/ --blocks 2 --nscans ${EN_NSCANS[2]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block3/ --blocks 3 --nscans ${EN_NSCANS[3]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block4/ --blocks 4 --nscans ${EN_NSCANS[4]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block5/ --blocks 5 --nscans ${EN_NSCANS[5]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block6/ --blocks 6 --nscans ${EN_NSCANS[6]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block7/ --blocks 7 --nscans ${EN_NSCANS[7]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block8/ --blocks 8 --nscans ${EN_NSCANS[8]} --output-dir=$OUTPUTDIR -- ${FEATURES}
python $SCRIPTSDIR/generate-regressors.py --input-dir $INPUTDIR/Block9/ --blocks 9 --nscans ${EN_NSCANS[9]} --output-dir=$OUTPUTDIR -- ${FEATURES}


