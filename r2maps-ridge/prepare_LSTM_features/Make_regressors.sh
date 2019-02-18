#! /bin/bash


BASEDIR=/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge/all_regressors/${LINGUA} 

op=$PWD

cd $BASEDIR

for b in 1 2 3 4 5 6 7 8 9;
do
    for f in block$b/*; do echo ln -s $f  $b_`basename $f`; done
done

cd $op

