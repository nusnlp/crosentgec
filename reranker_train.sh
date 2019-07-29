#!/bin/bash

set -e
set -x


if [ $# -lt 2 ]; then
    echo "Usage: `basename $0` <outputs_dir> <dev_data> <optional-gpu-num>"
    exit 1
fi

moses_dir=tools/mosesdecoder
if [ ! -e $moses_dir ] ; then
    echo "Moses not found at path: $moses_dir"
    echo "Set variable moses_dir to Moses SMT path"
    exit 1
fi

nbest_reranker=tools/nbest-reranker
test_data_dir=data/test

###############
# training
###############

fairseq_outputs_dir=$1
dev_set=$2
input_dev=$fairseq_outputs_dir/$dev_set.out
output_dir=$fairseq_outputs_dir/reranking.$dev_set
train_dir=$output_dir/training/

lm_file='models/ngramlm/cclm.trie'

mkdir -p $train_dir
echo "[weight]" > $train_dir/rerank_config.ini
echo "F0= 0.5" >> $train_dir/rerank_config.ini
echo "EditOps0= 0.2 0.2 0.2" >> $train_dir/rerank_config.ini
echo "LM0= 0.5" >> $train_dir/rerank_config.ini
echo "BERT0= 0.5" >> $train_dir/rerank_config.ini
echo "WordPenalty0= -1" >> $train_dir/rerank_config.ini

featstring="EditOps(name='EditOps0'), LM('LM0', '$lm_file', normalize=False), BERT(name='BERT0', cased=True, large=False), WordPenalty(name='WordPenalty0')"


########################
##### TRAINING #########
########################

# reformating the nbest file
python2.7 scripts/nbest_reformat.py -i $input_dev --debpe > $input_dev.mosesfmt

# # augmenting the dev nbest
if [ $# -eq 2 ]; then
    device=$3
else
    device=0
fi

CUDA_VISIBLE_DEVICES=$device python3 $nbest_reranker/augmenter.py -s $test_data_dir/$dev_set/$dev_set.tok.src -i $input_dev.mosesfmt -o $train_dir/$dev_set.moses-nbest.augmented.txt -f "$featstring"

# # training the nbest to obtain the weights
python3 $nbest_reranker/train.py -i $train_dir/$dev_set.moses-nbest.augmented.txt -r $test_data_dir/$dev_set/$dev_set.m2 -c $train_dir/rerank_config.ini --threads 12 --tuning-metric m2 --predictable-seed -o $train_dir --moses-dir $moses_dir --no-add-weight

cp $train_dir/weights.txt $output_dir/weights.$dev_set.txt

