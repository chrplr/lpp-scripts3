#!/bin/bash

# path to self
MYPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# input corpus (for example, path to frwiki.txt)
INPUT=$1
# directory where to leave a post-processed train/val/test split
OUTPUT_DIR=$2

# maximum sentence length
MAX_LENGTH=25 

echo tokenize and separate in sentences
if [ -f $INPUT ]; then
    java -cp ${MYPATH}/../src/CoreNLP/stanford-corenlp.jar edu.stanford.nlp.international.french.process.FrenchTokenizer < $INPUT > ${INPUT}.tok
fi
echo filtering and formatting
if [ -f ${INPUT}.tok ]; then
    < ${INPUT}.tok sed 's/ \(\.\|!\|\?\) / \1\n/g' | \
        perl -e "while(<>){if(scalar(split(' ')) < $MAX_LENGTH) {print};}" | \
        PERLIO=:utf8 perl -pe '$_=lc' | \
        grep -v '^\s*$' > ${INPUT}.tok.fmt
fi
echo compute word frequencies
<${INPUT}.tok.fmt awk '{a[$1]++}END{for(k in a)print a[k]"\t"k}' RS=" |\n"  | sort -nr > ${INPUT}.wfreq
echo extract a reduced set of the vocabulary
mkdir -p ${OUTPUT_DIR}
cat $MYPATH/stimuli-vocab.txt <(head -100000 ${INPUT}.wfreq | cut -f2) | sort -u > ${OUTPUT_DIR}/reduced-vocab.txt
sed -i '1i_UNK_' ${OUTPUT_DIR}/reduced-vocab.txt
sed -i '1i</s>' ${OUTPUT_DIR}/reduced-vocab.txt

echo shuffle and remove OOV words
shuf ${INPUT}.tok.fmt | python mask_with_unk.py ${OUTPUT_DIR}/reduced-vocab.txt > ${OUTPUT_DIR}/full.txt
echo splitting data
./split-data.sh ${OUTPUT_DIR}/full.txt

