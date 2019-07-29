#!/bin/bash

# This scripts prepares training and dev data for cross-sentence GEC.
# Author: shamil.cm@gmail.com, weiqi@u.nus.edu

set -e
set -x

if [ $# -ne 2 ]; then
	echo "Usage: `basename $0` <Lang8-file-path> <Nucle-file-path>"
fi

# paths to raw data files
LANG8V2=$1
NUCLE_TAR=$2

mkdir -p data/tmp
tmp_dir=data/tmp

# LANG-8 v2
#############
# Preparation of Lang-8 data
echo "[`date`] Preparing Lang-8 data... (NOTE:Can take several hours, due to LangID.py filtering...)" >&2
python3 scripts/lang8_preprocess.py --dataset $LANG8V2 --language English --id en --output $tmp_dir/lang-8-20111007-L1-v2.xml
python3 scripts/partition_data_into_train_and_dev.py --dataset $tmp_dir/lang-8-20111007-L1-v2.xml --train $tmp_dir/lang8-train.xml --dev $tmp_dir/lang8-dev.xml --limit 0
python2 scripts/sentence_pairs_with_ctx.py --train --tokenize --maxtokens 80 --mintokens 1 --input $tmp_dir/lang8-train.xml  \
	--src-ctx $tmp_dir/lang8.src-trg.ctx --src-src $tmp_dir/lang8.src-trg.src --trg-trg $tmp_dir/lang8.src-trg.trg


# NUCLE
#############
# Preparation of NUCLE data
echo "[`date`] Preparing NUCLE data..." >&2
tar -zxvf $NUCLE_TAR -C $tmp_dir/
nucle_dir=$tmp_dir/release3.2
python2 $nucle_dir/scripts/preprocess.py -l $nucle_dir/data/nucle3.2.sgml \
	$tmp_dir/nucle3.2-preprocessed.conll $tmp_dir/nucle3.2-preprocessed.conll.ann $tmp_dir/nucle3.2-preprocessed.conll.m2
python3 scripts/nucle_preprocess.py $tmp_dir/nucle3.2-preprocessed.conll $tmp_dir/nucle3.2-preprocessed.conll.m2 $tmp_dir/nucle3.2.xml
python3 scripts/partition_data_into_train_and_dev.py --dataset $tmp_dir/nucle3.2.xml \
	--train $tmp_dir/nucle-train.xml --dev $tmp_dir/nucle-dev.xml --limit 5000 --m2 $tmp_dir/nucle3.2-preprocessed.conll.m2 --dev-m2 $tmp_dir/nucle-dev.raw.m2
python2 scripts/sentence_pairs_with_ctx.py --train --maxtokens 80 --mintokens 1 --input $tmp_dir/nucle-train.xml \
	--src-ctx $tmp_dir/nucle.src-trg.ctx --src-src $tmp_dir/nucle.src-trg.src --trg-trg $tmp_dir/nucle.src-trg.trg
python2 scripts/sentence_pairs_with_ctx.py --dev --maxtokens 80 --mintokens 1 --input $tmp_dir/nucle-dev.xml \
	--src-ctx $tmp_dir/nucle-dev.src-trg.ctx --src-src $tmp_dir/nucle-dev.src-trg.src --trg-trg $tmp_dir/nucle-dev.src-trg.trg
python3 scripts/m2_preprocess.py --nucle-dev $tmp_dir/nucle-dev.src-trg.src --dev-m2 $tmp_dir/nucle-dev.raw.m2 --processed-m2 $tmp_dir/nucle-dev.preprocessed.m2


# preprocessed training and dev data
#############
mkdir -p data/processed
BPE_MODEL=models/bpe/mlconvgec_aaai18_bpe.model
out_dir=data/processed
for ext in ctx src trg; do
	cat $tmp_dir/lang8.src-trg.$ext > $tmp_dir/train.src-trg.$ext
	cat $tmp_dir/nucle.src-trg.$ext >> $tmp_dir/train.src-trg.$ext
	cp $tmp_dir/nucle-dev.src-trg.$ext $tmp_dir/valid.src-trg.$ext
	scripts/apply_bpe.py -c $BPE_MODEL < $tmp_dir/train.src-trg.$ext > $out_dir/train.src-trg.$ext
	scripts/apply_bpe.py -c $BPE_MODEL < $tmp_dir/valid.src-trg.$ext > $out_dir/valid.src-trg.$ext
done
cp $tmp_dir/nucle-dev.preprocessed.m2 data/nucle-dev.preprocessed.m2

rm -rf $tmp_dir

# copy the dictionary
BASEURL=https://tinyurl.com/yd6wvhgw/mlconvgec2018/models
curl -L -o $out_dir/dict.ctx.txt $BASEURL/data_bin/dict.src.txt
curl -L -o $out_dir/dict.src.txt $BASEURL/data_bin/dict.src.txt
curl -L -o $out_dir/dict.trg.txt $BASEURL/data_bin/dict.trg.txt
