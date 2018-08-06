#!/usr/bin/env python
import argparse
import pickle

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('corpus')
    ap.add_argument('-r', '--raw', action='store_true', 
            help='Extract sentences')
    ap.add_argument('-l', '--length', type=int)

    args = ap.parse_args()

    corpus = pickle.load(open(args.corpus, 'rb'))

    for sentence, meta, data in corpus:
        if args.length and len(sentence) != args.length:
            continue
        #print(' '.join(sentence))
        #if not args.raw:
            #for k in meta:
            #    print('### '+ k +  " " + str(meta[k]))
            #for k in data:
            #    print('*** '+ k +  " " + " ".join(map(str, data[k])))
        print(tree_to_brackets(eval(meta['tree'])))

def tree_to_brackets(p):
    is_leaf, tag, data = p
    if is_leaf:
        return "X"
    else:
        return "[{}]".format("".join(tree_to_brackets(st) for st in data))

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
