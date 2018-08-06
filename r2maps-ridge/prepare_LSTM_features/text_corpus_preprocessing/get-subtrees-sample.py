#!/usr/bin/env python
import argparse
import pickle
import re
import random
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sentences')
    ap.add_argument('-n', '--n-samples', default=1000, type=int)
    ap.add_argument('-p', '--pos-name', default='AD[JV]P')
    ap.add_argument('-t', '--patterns', action='append')
    ap.add_argument('-o', '--output')

    
    args = ap.parse_args()

    patterns = args.patterns
    #patterns = ["RRB", "LLB", "LRB", "RLB", "BB"]
    
    sentences = pickle.load(open(args.sentences, 'rb'))

    pat2sen = {k: [] for k in patterns}
    pat2tags_hist = {k: Counter() for k in patterns}

    random.shuffle(sentences)
    # let's try with just picking sentences that contain a single occurrence
    # of a tree
    for s in sentences:
        tokenized, meta, features = s
        sen_pats = get_patterns(meta, patterns)
        if len(sen_pats) == 1:
            pat_type, idx, pos, tags= sen_pats[0]
            if len(pat2sen[pat_type]) >= args.n_samples:
                continue
            if not re.search(args.pos_name, pos):
                continue
            pat2sen[pat_type].append((tokenized,idx, tags))
            for t in tags.split("_")[1:3]:
                pat2tags_hist[pat_type][t] += 1
    data = []
    for k,v in pat2sen.items():
        for sent, pos, tags in pat2sen[k]:
            data.append(({'pattern': k, 'sentence': sent, 'tree-pos': pos, 'tags': tags}))
            print(" ".join(sent))
    if args.output:
        with open(args.output, 'wb') as fout:
            pickle.dump(data, fout)

def get_patterns(meta, patterns):
    pats = []
    for p in patterns:
        k = "subtree_{}".format(p)
        if len(meta[k]) == 0:
            continue
        for occurrence in meta[k].split(" "):
            idx, pos,tags = occurrence.split(":",2)
            pats.append((p,int(idx),pos,tags))
    return pats

main()


