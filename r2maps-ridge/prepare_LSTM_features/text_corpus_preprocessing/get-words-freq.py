#!/usr/bin/env python
import argparse
import fileinput

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('freqs')
    ap.add_argument('words')
    args = ap.parse_args()

    freqs = {}
    for l in open(args.freqs):
        ls = l.rstrip('\n').split('\t')
        freqs[ls[1]] = int(ls[0])

    for l in open(args.words):
        w = l.rstrip('\n')
        print("{}\t{}".format(w, freqs.get(w, 0)))
main()
