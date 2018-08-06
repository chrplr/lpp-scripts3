#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import scipy.stats as stats

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('corpus')
    ap.add_argument('-r', '--raw', action='store_true', 
            help='Extract sentences')
    ap.add_argument('-l', '--length', action='store_true')

    args = ap.parse_args()

    corpus = pickle.load(open(args.corpus, 'rb'))

    if args.length:
        print(len(corpus))
        return

    for i in range(10):
        depth = []
        length = []
        for sentence, meta, data in corpus:
            length.append(list(range(i, len(sentence))))
            depth.append(data['all'][i:])
        depth = np.concatenate(depth)
        length = np.concatenate(length)
        print(i, "{:.2f}".format(stats.pearsonr(depth, length)[0]), len(depth))
            #print(format_constituent(eval(meta['tree'])))

def format_constituent(p):
    t = p[1]
    if p[0]:
        return "({} {})".format(p[1], p[2])
    else:
        dont_indent = all(sp[0] for sp in p[2]) # dont indent if all subconstituens are leafs
        ret = "({} ".format(t)
        if not dont_indent:
            ret += '\n'
        if dont_indent:
            ret += " ".join(format_constituent(sp) for sp in p[2])
        else:
            for sp in p[2]:
                for l in format_constituent(sp).split("\n"):
                    ret += "  {}\n".format(l)
        ret += ")"
        return ret

main()
