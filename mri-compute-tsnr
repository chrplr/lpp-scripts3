#! /usr/bin/env python
# Time-stamp: <2017-06-05 09:11:22 cp983411>

import sys
import os

from nipype.algorithms.confounds import TSNR

tsnr = TSNR()

for f in sys.argv[1:]:
    print("Processing %s..." %f)
    tsnr.inputs.in_file = f
    fname, ext = os.path.splitext(f)
    tsnr.inputs.tsnr_file = fname + "_tsnr.nii.gz"
    tsnr.inputs.mean_file = fname + "_mean.nii.gz"
    tsnr.inputs.detrended_file = fname + "_detrended.nii.gz"
    tsnr.inputs.stddev_file = fname + "_stddev.nii.gz"
    tsnr.inputs.regress_poly = 4
    tsnr.run()


