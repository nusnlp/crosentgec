#!/bin/bash

set -e
set -x


if [ $# -lt 3 ]; then
    echo "Usage: `basename $0` <outputs_dir> <test_data> <weights> <optional-gpu-num>"
    exit 1
fi


nbest_reranker=tools/nbest-reranker
test_data_dir=data/test


fairseq_outputs_dir=$1
test_set=$2
weights_file=$3
input_test=$fairseq_outputs_dir/$test_set.out
output_dir=$fairseq_outputs_dir/reranked/

mkdir -p $output_dir
lm_file='models/ngramlm/cclm.trie'
featstring="EditOps(name='EditOps0'), LM('LM0', '$lm_file', normalize=False), BERT(name='BERT0', cased=True, large=False), WordPenalty(name='WordPenalty0')"


########################
##### TESTING ##########
########################

# reformating the nbest file
python2.7 scripts/nbest_reformat.py -i $input_test --debpe > $input_test.mosesfmt

if [ $# -eq 4 ]; then 
    device=$4
else
    device=0
fi
CUDA_VISIBLE_DEVICES=$device python3 $nbest_reranker/augmenter.py -s $test_data_dir/$test_set/$test_set.tok.src -i $input_test.mosesfmt -o $output_dir/$test_set.moses-nbest.augmented.txt -f "$featstring"

python3 $nbest_reranker/rerank.py -i $output_dir/$test_set.moses-nbest.augmented.txt  -w $weights_file -o $output_dir --clean-up

mv $output_dir/$test_set.moses-nbest.augmented.txt.reranked.1best $output_dir/$test_set.out.txt


