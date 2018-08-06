#!/usr/bin/env python

# This script takes a  corpus parsed with Satanford's CoreNLP and produces
# a time series of activations that can later be used for correlating 
# individual units activations.

import argparse
import fileinput
import pickle
import sys
import os
import traceback
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--vocab', required=True, help='We only care about'
    ' sentences for which we have all words in this vocabulary')
    ap.add_argument('-m', '--mode', choices=['tokenize', 'depth', 'subtrees'])
    ap.add_argument('-o', '--output')
    ap.add_argument('--silent', default=False ,action='store_true')

    args = ap.parse_args()

    vocab = set(l.rstrip('\n') for l in open(args.vocab))

    all_features = []
    skipped = 0
    total = 0
    f = open(args.data)
    f.seek(0, os.SEEK_END)
    fsize = f.tell()
    f.seek(0, os.SEEK_SET)
    pbar = tqdm(total=fsize)
    before = f.tell()
    for sentence in sentences(f, remove_unary=True):
        pbar.update(f.tell()-before)
        before = f.tell()
        #sys.stderr.write("Complete: {:.2f}%\r".format(f.tell()/fsize*100))
        tokenized_sentence = tokens(sentence)
        total += 1
        skip = False
        for w in tokenized_sentence:
            if w not in vocab:
                skip = True
                skipped += 1
                break
        if not skip and len(tokenized_sentence) > 60:
            skip = True
            skipped +=1
        elif not skip:
            if not args.silent:
                print(" ".join(tokens(sentence)))
            if not args.mode == 'tokenize':
                meta = {}
                meta["head-type"]  = get_head_type(sentence['phrase-parse'])
                meta["tree"] = sentence['phrase-parse']
                if args.mode == 'subtrees': 
                    for pat_name, pattern in patterns.items():
                        meta["subtree_{}".format(pat_name)] = " ".join(
                                map(lambda t: "{}:{}:{}".format(*t), 
                                    tree_find(sentence['phrase-parse'], pattern)))
                features = {}
                if args.mode == 'depth':
                    for k,feature_extractor in {"first_NP": first_NP_depth, 
                            "first_VP": first_VP_depth, 
                            "top_NP": top_NP_depth,
                            "top_VP": top_VP_depth, 
                            "all": open_constituents}.items():
                            #closed_constituents]:
                        features[k] = feature_extractor(sentence['phrase-parse'])
                        diff_k = 'diff_' + k
                        features[diff_k] = differences(features[k])
                        features['pos_' + diff_k] = filter_features(
                                features[diff_k], lambda f: f >= 0)
                        features['neg_' + diff_k] = filter_features(
                                features[diff_k], lambda f: f <= 0)
                if not args.silent:
                    for k in meta:
                        print('### '+ k +  " " + str(meta[k]))
                    print(format_constituent(meta['tree']))
                    for k in features:
                        print('*** '+ k +  " " + " ".join(map(str, features[k])))
                all_features.append((tokenized_sentence, meta, features))
    pbar.close()
    sys.stderr.write("Skipped sentences: {} / {}".format(skipped, total))
    if args.output:
        pickle.dump(all_features, open(args.output, 'wb'))

patterns = {
        "RRB": "[ X [ X [ X X ] ] ]",
        "LLB": "[ [ [ X X ] X ] X ]",
        "LRB": "[ [ X [ X X ] ] X ]",
        "RLB": "[ X [ [ X X ] X ] ]",
        "BB": "[ [ X X ] [ X X ] ]"
    }

def differences(features):
    d = []
    for i,f in enumerate(features):
        if i > 0 and features[i-1] != "-" and f != "-":
            d.append(f - features[i-1])
        else:
            d.append("-")
    return d

def filter_features(features, condition):
    d = [] 
    for f in features:
        if f != "-" and condition(f):
            d.append(f)
        else:
            d.append("-")
    return d

def tree_find(tree, pattern):
    '''Returns the first occurrence of the pattern in the tree. The
    patter is e.g. of the shape [NP X [ X [ X X ] ] ]'''
    flat_tree = tree_to_sequence(tree)
    pattern = pattern.strip().split(" ")
    positions = []
    word_pos = 0
    pat_length = sum([1 for x in pattern if x=="X"])
    for i in range(len(flat_tree)):
        tok = flat_tree[i]
        if not tok.startswith('[') and not tok.startswith(']'):
            word_pos += 1
        matches = True
        for j in range(len(pattern)):
            if not pattern_match(flat_tree[i+j], pattern[j]):
                matches = False
                break
        if matches:
            pos_tags = [tag for tag in flat_tree[i:] if tag[0] not in '[]'][:pat_length]
            positions.append((word_pos, tok[1:], "_".join(pos_tags)))
    return positions

def pattern_match(chunk, pattern):
    if pattern.startswith("["):
        if pattern == "[":
            return chunk.startswith("[")
        else:
            # match node tag
            return chunk[1:] == pattern[1:]
    elif pattern == "]":
        return chunk == "]"
    elif pattern == "X":
        # variable
        return pattern.isalpha()
    else:
        return pattern == chunk



def tree_to_sequence(tree):
    is_leaf, tag, payload = tree
    if is_leaf:
        return [tag]
    else:
        return ["[" + tag ] + sum((tree_to_sequence(st) for st in payload), []) + ["]"]

def chomp(it):
    return it.readline().rstrip('\n')

def tokens(sentence):
    return [t['Text'] for t in sentence['tokens']]

def first_NP_depth(tree):
    return first_NT_depth(tree, "NP")[0]

def first_VP_depth(tree):
    return first_NT_depth(tree, "VP")[0]

def top_NP_depth(tree):
    return NT_depth(tree, "NP")

def top_VP_depth(tree):
    return NT_depth(tree, "VP")

def open_constituents(tree):
    return NT_depth(tree, "S")

def _remove_unary(tree):
    is_leaf, tag, payload = tree
    if not is_leaf and len(payload) == 1:
        # ignore this tree
        return _remove_unary(payload[0])
    elif is_leaf:
        return tree
    else:
        return (is_leaf, tag, [_remove_unary(st) for st in payload])

def NT_depth(tree, target, distance_to_NT=None):
    is_leaf, tag, payload = tree
    if is_leaf:
        if distance_to_NT is None:
            return ["-"]
        else:
            return [distance_to_NT]
    else:
        if tag == target and distance_to_NT is None:
            distance_to_NT = 0
        elif distance_to_NT is not None:
            distance_to_NT += 1
        return sum((NT_depth(st, target, distance_to_NT) for st in payload), [])

def first_NT_depth(tree, target, distance_to_NT=None, found=False):
    is_leaf, tag, payload = tree
    if is_leaf:
        if distance_to_NT is None:
            return ["-"], found
        else:
            return [distance_to_NT], found
    else:
        if tag == target and distance_to_NT is None and not found:
            distance_to_NT = 0
            found = True
        elif distance_to_NT is not None:
            distance_to_NT += 1
        ret = []
        for st in payload:
            st_seq, st_found = first_NT_depth(st, target, distance_to_NT, found)
            found = found or st_found
            ret.extend(st_seq)
        return ret, found

def get_head_type(tree):
    return "_".join(flat_tags(tree, 1))

def flat_tags(tree, level=None):
    is_leaf, tag, subtrees = tree
    if is_leaf:
        return[]
    if level == 0:
        return [tag]
    else:
        disc_level = level-1 if level is not None else None
        return [tag]  + sum((flat_tags(st, disc_level) for st in subtrees), [])
    

def sentences(fin, remove_unary=False):
    fin.seek(0, os.SEEK_END)
    fsize = fin.tell()
    fin.seek(0, os.SEEK_SET)
    it = fin
    line = chomp(it)
    n = 1
    while True:
        try:
            if line.startswith("Document: ID="):
                line = chomp(it)
                n += 1
            if it.tell() == fsize:
                return
            assert line.startswith("Sentence"), "Sentence at line {} unexpected: {}".format(n, line)
            line = chomp(it)
            n += 1
            sent = {}
            sent['original'], line = get_original(it, line)
            sent['tokens'], line = get_tokens(it, line)
            sent['phrase-parse'], line = get_phrase_parse(it, line)
            if remove_unary:
                sent['phrase-parse'] = _remove_unary(sent['phrase-parse'])
            sent['dep-parse'], line = get_dep_parse(it, line)
            assert len(line) == 0
            if len(sent['original']) == 1:
                # we filter out multi-line sentences as they are most likely wrong
                yield sent
            line = chomp(it)
        except KeyboardInterrupt:
            raise
        except BaseException as ex:
            print("Error at position {} just before line: \n{}"\
                    .format(it.tell(), chomp(it)), file=sys.stderr)
            traceback.print_exc()
            # recovering routine
            while not line.startswith("Sentence"):
                if it.tell() == fsize:
                    return
                line = chomp(it)

def get_original(it, line):
    original = []
    while not line.startswith('[Text='):
        original.append(line)
        line = chomp(it)
    return original, line

def get_tokens(it, line):
    tokens = []
    assert line[0] == '['
    while line[0] == '[':
        token_parts = line[1:-1].split(" ")
        token = {}
        for part in token_parts:
            k,v = part.split("=", 1)
            token[k] = v
        tokens.append(token)
        line = chomp(it)
    return tokens, line

def get_phrase_parse(it, line):
    assert line[0] == '('
    parse_text = []
    while len(line) > 0:
        parse_text.append(line)
        line = chomp(it)
    return parse("".join(parse_text)), line

def get_dep_parse(it, line):
    # dummy stub
    line = chomp(it)
    while len(line) > 0:
        line = chomp(it)
    return None, line

def parse(l):
    #this is a constituent
    assert l[0] == '(' and l[-1] == ')',l
    subconstituents = []
    c = 0
    constituent_start = None
    for i in range(1, len(l)-1):
        if l[i] == '(':
            if c == 0:
                constituent_start = i
            c+=1
        elif l[i] == ')':
            assert constituent_start is not None
            c-=1
            if c == 0:
                subconstituents.append((constituent_start, i+1))
                constituent_start = None

    tag=l[1:l.index(" ")]
    if subconstituents:
        # non terminal
        return (False, tag, [parse(l[b:e]) for b,e in subconstituents])
    else:
        # terminal
        return (True, tag, l[l.index(" ")+1:-1])


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
