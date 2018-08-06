import sys

vocab_fn = sys.argv[1]
try:
    corpus = open(sys.argv[2])
except:
    corpus = sys.stdin

vocab = set(l.rstrip('\n') for l in open(vocab_fn))
unk = "_UNK_"

total = 0
invocab = 0
for l in corpus:
    newl = []
    if len(l.strip()) == 0:
        continue
    sent = l.rstrip('\n').split(" ")
    if len(sent) <= 2:
        continue
    sent_unks = sum(1 if w not in vocab else 0 for w in sent)
    # also filters sentences that have more than 5% unknowns
    if sent_unks > 0.05 * len(sent):
        continue
    for w in sent:
        if w in vocab:
            newl.append(w)
            invocab += 1
        else:
            newl.append(unk)
        total += 1 
    print(" ".join(newl))
    #if total > 1e8:
    #    break
print("Total words: {}".format(total), file=sys.stderr)
print("In vocabulary: {:.2f}%".format(float(invocab)/total*100), file=sys.stderr)
print("OOV: {:.2f}%".format(100 - float(invocab)/total*100), file=sys.stderr)

