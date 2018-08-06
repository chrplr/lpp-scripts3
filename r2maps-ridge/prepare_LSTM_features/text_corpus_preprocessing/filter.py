#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import argparse
import math
import numpy as np
import pickle
import sys
import os
import random
import operator 
import pickle
import scipy.stats
from tqdm import tqdm

from collections import defaultdict
np.seterr(all='raise')


def main():
    ap = argparse.ArgumentParser(description="""
    """)
    ap.add_argument('corpus')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--min-length', type=int)
    ap.add_argument('--max-length', type=int)
    ap.add_argument('--head-type', )
    ap.add_argument('--decorrelate', action='store_true', default=False)
    ap.add_argument('--max-size', type=int)

    args = ap.parse_args()

    print('Loading corpus...')
    corpus = pickle.load(open(args.corpus, 'rb'))

    # lets start by shuffling things aroung
    random.shuffle(corpus)

    print('Corpus size:', len(corpus))
    filtered_corpus = []
    for i,t in enumerate(tqdm(corpus, desc='Filtering by meta-data')):
        sentence, meta, data = t

        # do filtering:
        if args.min_length and len(sentence) < args.min_length:
            continue
        
        if args.max_length and len(sentence) > args.max_length:
            continue

        if args.head_type and meta['head-type'] != args.head_type:
            continue

        filtered_corpus.append(t)

    print('Filtered:', len(corpus) - len(filtered_corpus))
    corpus = filtered_corpus

    if args.decorrelate:
        start = 5
        depth_series, length_series = zip(
                *[(np.array(data['all'][start:]), 
                    np.array(list(range(start, len(sentence)))))
                for sentence, meta, data 
                in tqdm(corpus, desc='Collecting depth&length')])
        print("Pre-procedure correlation R={:.2f} p={:.2e}".format(
                *scipy.stats.spearmanr(
                    np.concatenate(depth_series), 
                    np.concatenate(length_series))))
        print('Preparing data for decorrelation...')
        x1 = [np.max(vals) for vals in length_series]
        x2 = [np.max(vals) for vals in depth_series]
        m_1, s_1, q10_1, q90_1 = np.mean(x1), 3*np.std(x1), \
                np.percentile(x1, 10), np.percentile(x1, 90)
        m_2, s_2, q10_2, q90_2 = np.mean(x2), 3*np.std(x2), \
                np.percentile(x2, 10), np.percentile(x2, 90)
        decorr = BivariateDecorrelationFilter(
                m_1, s_1, (q10_1, q90_1),
                m_2, s_2, (q10_2, q90_2))

        decorr_idxs = []
        for i in tqdm(range(len(x1)), desc='Decorrelation filter'):
            if decorr.filter((x1[i],x2[i])):
                decorr_idxs.append(i)

        depth_series = [depth_series[i] for i in decorr_idxs]
        length_series = [length_series[i] for i in decorr_idxs]
        print("Post-procedure correlation R={:.2f} p={:.2e}".format(
                *scipy.stats.spearmanr(
                    np.concatenate(depth_series), 
                    np.concatenate(length_series))))

        filtered_corpus = [corpus[i] for i in decorr_idxs]
        print('Filtered:', len(corpus) - len(filtered_corpus))
        corpus = filtered_corpus

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as fout:
        pickle.dump(corpus, fout)


class BivariateDecorrelationFilter():
    def __init__(self, m1, s1, lim1, m2, s2, lim2, grow_size=100):
        cv = np.array([[s1, 0], [0, s2]])
        mean = np.array([m1, m2])

        Z = 0
        self.target_distr = {}
        for i in range(int(lim1[0]), int(lim1[1])):
            for j in range(int(lim2[0]), int(lim2[1])):
                self.target_distr[(i,j)] = \
                        scipy.stats.multivariate_normal.pdf(
                                [i,j], mean=mean, cov=cv)

        self.max_items = 0
        self.max_items_delta = int(lim2[1] - lim2[0]) * \
                               int(lim1[1] - lim1[0]) * grow_size

        self.grow_max_items()
    
    def grow_max_items(self):
        self.max_items += self.max_items_delta

        # recompute bin sizes
        self.bin_targets = {}
        Z = sum(self.target_distr.values())
        for k,v in self.target_distr.items():
            self.bin_targets[k] = v/Z * self.max_items

    def filter(self, vals):
        if vals in self.bin_targets and self.bin_targets[vals] > 0:
            self.bin_targets[vals] -= 1
            return True
        return False


main()
