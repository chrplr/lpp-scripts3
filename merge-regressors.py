#! /usr/bin/env python
# Time-stamp: <2017-07-20 17:10:17 cp983411>

""" read a series of regressor files, containing each 1 numeric column
and paste them horizontally to create 9 matrices in csv file format.
On the comand line list the name of the vectors; 
The filenames must have the pattern '[1-9]_%s_reg.csv' where %S is the name of the vector """

import sys
import os
import getopt
import os.path as op
import pandas as pd

NRUNS=9

# parse command line
try:
    opts, args = getopt.getopt(sys.argv[1:],
                               "i:o:",
                               ["input_dir=", "output_dir="])

except getopt.GetoptError as err:
    print(err)
    sys.exit(2)
        
for o, a in opts:
    if o in ('-i', '--input_dir'):
        input_dir = a
    elif o in ('-o', '--output_dir'):
        output_dir = a
                        
if not op.isdir(output_dir):
    os.mkdir(output_dir)

# merge columns into matrices

for run in range(1, NRUNS + 1):
    a = []
    for s in args:
        fn = op.join(input_dir, "%s_%s_reg.csv" % (run, s))
        a.append(pd.read_csv(fn, header=None))
    merge = pd.concat(a, axis=1)
    merge.columns = args
    fn = op.join(output_dir, "dmtx_{}.csv".format(run))
    merge.to_csv(fn, index=False)
        
