#! /usr/bin/env python
# Time-stamp: <2018-04-07 14:26:31 cp983411>

"""Generate a fMRI regressors hrf from "event" files"""

import sys
import os.path as op
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from nistats.hemodynamic_models import compute_regressor
from multiprocessing import Pool

# import matplotlib.pyplot as plt

def onset2reg(df, nscans, tr):
    """ df : pandas dataframe with columnes onset and amplitude, and, optionnaly, duration
        nscans: number of scans
        tr : sampling period of scanning""" 
    n_events = len(df)
    onsets = df.onset
    amplitudes = df.amplitude
    if 'duration' in df.columns:  # optionaly, use "duration"
        durations = df.duration
    else:
        durations = np.zeros(n_events)

    conditions = np.vstack((onsets, durations, amplitudes))

    x = compute_regressor(exp_condition = conditions,
                          hrf_model = "spm",
                          frame_times = np.arange(0.0, nscans * tr, tr),
                          oversampling = 10)

    return pd.DataFrame(x[0], columns=['hrf'])


def process_onefile(csvfile, tr, nscans, overwrite=False, output_dir=None):

    fname, ext = op.splitext(csvfile)
    
    regfname = "%s_reg.csv" % op.basename(fname)
    if not output_dir is None:
        regfname = op.join(output_dir, regfname)
    if not(overwrite):
        if op.isfile(regfname):
            if op.getmtime(csvfile) < op.getmtime(regfname):
                print('Warning: %s not processed because %s exists and is more recent' % (csvfile, regfname))
            return
    df = pd.read_csv(csvfile)
    x = onset2reg(df, nscans, tr)
    x.to_csv(regfname, index=False, header=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generate fMRI regressors from 'events' files (i.e., csv files with 2 columns: onset and amplitude)")
    parser.add_argument("--tr", type=int, default=None,
                        help="TR in sec (sampling period for scans)")
    parser.add_argument("--nscans", type=int, default=None,
                        help="number of scans (= # of time-points in the output)")
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.set_defaults(overwrite=False)
    parser.add_argument('csv_file',
                nargs='+',
                action="append",
                default=[])
    
    args = parser.parse_args()

    tr = args.tr
    nscans = args.nscans
    overwrite = args.overwrite
    output_dir = args.output_dir
    
    for f in args.csv_file[0]:
        process_onefile(f, tr, nscans, overwrite, output_dir)
