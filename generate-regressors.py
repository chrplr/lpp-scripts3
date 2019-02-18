#! /usr/bin/env python
#! Time-stamp: <2018-04-08 10:06:29 cp983411>

""" take a list of predictor names on the command line and compute the hrf convolved regressors from the corresponding onset files [1-0]_name.csv """

import sys
import os
import os.path as op
from events2reg import process_onefile
from joblib import Parallel, delayed

import argparse

parser = argparse.ArgumentParser(description="Generate fMRI regressors from [1-9]_REG.csv onsets files)")
parser.add_argument("--output-dir", type=str, default='.')
parser.add_argument("--input-dir", type=str, default='.')
parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')
parser.add_argument('--overwrite', dest='overwrite', action='store_true')
parser.set_defaults(overwrite=False)
parser.add_argument('regressors',
                    nargs='+',
                    action="append",
                    default=[])
parser.add_argument('--nscans', nargs='+', action="append", default=[])
parser.add_argument('--blocks', nargs='+', action="append", default=[])
parser.add_argument('--lingua', type=str)

assert len(args.nscans) == len(args.blocks), 'number of scans is not equal to number of blocks'

args = parser.parse_args()
regressors = args.regressors[0]

if args.lingua == 'en' and not args.nscans:
    nscans = [282, 298, 340, 303, 265, 343, 325, 292, 368]  # numbers of scans in each session
elif args.lingua == 'fr' and not args.nscans:
    nscans = [309, 326, 354, 315, 293, 378, 332, 294, 336]  # numbers of scans in each session

parameters = [('%d_%s.csv' % (1 + session, reg), ns)
              for reg in regressors for (session, ns) in enumerate(nscans)]

Parallel(n_jobs=-2)(delayed(process_onefile) \
                    (op.join(args.input_dir, filen), 2.0, ns, args.overwrite, args.output_dir) for (filen, ns) in parameters)

