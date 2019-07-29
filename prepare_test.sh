#!/bin/bash

# This scripts prepares test data for cross-sentence GEC.

set -e
set -x

mkdir -p data/tmp
tmp_dir=data/tmp

# CoNLL-2014
#############
# downloading test data files
wget http://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz -O $tmp_dir/conll14st.tar.gz

# uncompressing the files
tar -zxvf $tmp_dir/conll14st.tar.gz -C $tmp_dir/
test_dir=$tmp_dir/conll14st-test-data

mkdir -p data/test/conll14st-test
out_dir=data/test/conll14st-test
python2 $test_dir/scripts/preprocess.py -l $test_dir/noalt/official-2014.0.sgml \
	$tmp_dir/conll14st-test.0.conll $tmp_dir/conll14st-test.0.conll.ann $tmp_dir/conll14st-test.0.conll.m2
python3 scripts/nucle_preprocess.py $tmp_dir/conll14st-test.0.conll $tmp_dir/conll14st-test.0.conll.m2 $tmp_dir/conll14st-test.0.xml
python2 scripts/sentence_pairs_with_ctx.py --test --input $tmp_dir/conll14st-test.0.xml \
	--src-ctx $out_dir/conll14st-test.tok.ctx --src-src $out_dir/conll14st-test.tok.src --trg-trg $tmp_dir/conll14st-test.0.tok.trg
cp $test_dir/noalt/official-2014.combined.m2 $out_dir/conll14st-test.m2


# CoNLL-2013
#############
wget https://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz -O $tmp_dir/conll13st.tar.gz

tar -zxvf $tmp_dir/conll13st.tar.gz -C $tmp_dir/
test_dir=$tmp_dir/release2.3.1/original/data

mkdir -p data/test/conll13st-test
out_dir=data/test/conll13st-test
python3 scripts/nucle_preprocess.py $test_dir/official-preprocessed.conll $test_dir/official-preprocessed.m2 $tmp_dir/conll13st-test.xml
python2 scripts/sentence_pairs_with_ctx.py --test --input $tmp_dir/conll13st-test.xml \
	--src-ctx $out_dir/conll13st-test.tok.ctx --src-src $out_dir/conll13st-test.tok.src --trg-trg $tmp_dir/conll13st-test.0.tok.trg
cp $test_dir/official-preprocessed.m2 $out_dir/conll13st-test.m2

rm -rf $tmp_dir
