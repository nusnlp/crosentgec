set -e
set -x

if [ $# -lt 3 ]; then
    echo "Usage: `basename $0` <test-data> <model-path>/<models-dir for ensemble> <dict-dir>  <optional gpu-number> "
    exit 1
fi

testset=$1
model_path=$2
DATA_DIR=$3

if [[ -d "$model_path" ]]; then
    models=`find -L $model_path/ -name "checkpoint_best.pt" -not -type d | tr '\n' ' ' | sed "s| \([^$]\)|:\1|g"`
	OUT_DIR=$model_path/outputs
elif [[ -f "$model_path" ]]; then
    models=$model_path
	OUT_DIR=`dirname $models`/outputs.`basename $model_path`
elif [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path"
fi

mkdir -p $OUT_DIR

# setting paths
FAIRSEQPY=fairseq
APPLYBPE=scripts/apply_bpe.py
BPE_MODEL=models/bpe/mlconvgec_aaai18_bpe.model

# setting input file path
INPUTFILE=./data/test/$testset/$testset.tok.src
CONTEXTFILE=./data/test/$testset/$testset.tok.ctx
REF=./data/test/$testset/$testset.m2

OUTPUT=$OUT_DIR/$testset.out
LOGFILE=$OUT_DIR/decode.$testset.log

TMP_DIR=$OUT_DIR/tmp.$testset$suffix.`date +%s`
mkdir $TMP_DIR

echo "testset: $testset" | tee -a $LOGFILE
echo "model_path: $models" | tee -a $LOGFILE
echo "processed_dir: $DATA_DIR" | tee -a $LOGFILE

python2.7 $APPLYBPE -c $BPE_MODEL < $INPUTFILE > $TMP_DIR/input.src
python2.7 $APPLYBPE -c $BPE_MODEL < $CONTEXTFILE > $TMP_DIR/input.ctx

if [ $# -lt 4 ] ; then
    DEVICE=0
else
    DEVICE=$4
fi

beam=12
threads=12
nbest=$beam


echo "START TIME:`date`" | tee -a $LOGFILE

CUDA_VISIBLE_DEVICES=$DEVICE python $FAIRSEQPY/interactive_multi.py --no-progress-bar --path $models --beam $beam --nbest $beam --replace-unk --source-lang src --target-lang trg --input-files $TMP_DIR/input.src $TMP_DIR/input.ctx --num-shards $threads --task translation_ctx  $DATA_DIR  > $OUTPUT


cat $OUTPUT | grep "^H"  | python3 -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if i%$nbest == 0 ]); print(x)" | cut -f3 | sed 's|@@ ||g' | sed '$ d' > $OUTPUT.txt
echo "END TIME:`date`" | tee -a $LOGFILE


rm -r $TMP_DIR
