#!/usr/bin/env python
import subprocess
import argparse
import itertools
import json
import sys

parser = argparse.ArgumentParser(description='Create a grid from json')
parser.add_argument('filename', help='Json or python that contains the grid parameters')
parser.add_argument('--cmd', required=True)
parser.add_argument('--project', required=True)
parser.add_argument('--experiment', required=True)
parser.add_argument('--run', action='store_true', default=False)

args = parser.parse_args()
with open(args.filename, 'r') as f:
    if args.filename.endswith('.json'):
        grid = json.load(f)
    else:
        grid = eval(f.read())

project = args.project
experiment= args.experiment
namefmt = grid['name']
parameters = grid['parameters']

perms = list(itertools.product(*parameters.values()))

names = set()
for p in perms:
    argstr = ""
    name = namefmt.format(**dict(zip(parameters.keys(),
                          [str(p_i).replace('/', '~').replace('-', '_') for p_i in p])))
    for i,k in enumerate(parameters.keys()):
        if type(p[i]) == bool:
            if p[i]:
                argstr += " --" + str(k)
        else:
            argstr += " --" + str(k) + " " + str(p[i]).format(name=name)
    if name in names:
        sys.stderr.write('WARNING: {} already exists\n'.format(name))
    else:
        names.add(name)
    cmd = "PROJECT={project} EXPERIMENT={experiment} NAME={name} {cmd} {argstr}".format(
    #cmd = "sbatch --partition=learnfair --nodes=1 --ntasks-per-node=1 --job-name={name} " \
    #    "--output=checkpoint-data/%j.out --error=checkpoint-data/%.err --gres=gpu:1 --cpus-per-task=4 " \
    #    "{cmd} {argstr}".format(
        project=project,
        experiment=experiment,
        name=name,
        cmd=args.cmd,
        argstr=argstr)
    if args.run:
        subprocess.run(cmd, shell=True)
    else:
        print(cmd)
