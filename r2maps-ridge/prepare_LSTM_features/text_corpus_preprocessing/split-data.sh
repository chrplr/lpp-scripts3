#!/bin/bash
P=$(dirname $1)
split -l $[ $(wc -l $1|cut -d" " -f1) * 99 / 100 ] $1
mv xaa $P/train.txt
mv xab val-test.txt
split -l $[ $(wc -l val-test.txt|cut -d" " -f1) * 50 / 100 ] val-test.txt
mv xaa $P/valid.txt
mv xab $P/test.txt
rm val-test.txt
rm -rf xac
