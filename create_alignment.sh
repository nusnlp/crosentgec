set -e
set -x

if [ $# -ne 1 ]; then
    echo "Usage: `basename $0` <processed-data-dir>"
    exit 1
fi
MOSESDECODER_DIR=$PWD/tools/mosesdecoder
FAIRSEQPY=$PWD/fairseq
FAST_ALIGN_BUILD=$PWD/tools/fast_align/build/
PROCESSED_DIR=$1

python3 $FAIRSEQPY/scripts/build_sym_alignment.py --fast_align_dir $FAST_ALIGN_BUILD --mosesdecoder_dir $MOSESDECODER_DIR --source_file $PROCESSED_DIR/train.src-trg.src --target_file $PROCESSED_DIR/train.src-trg.trg --output_dir $PROCESSED_DIR/alignment
