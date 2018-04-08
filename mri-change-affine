#! /usr/bin/env python

import sys
import nibabel as nib
from nilearn.image import resample_to_img


ref_img = nib.load(sys.argv[1])
src_img = nib.load(sys.argv[2])

tgt_img = resample_to_img(src_img, ref_img)

nib.save(tgt_img, 'target.nii')
