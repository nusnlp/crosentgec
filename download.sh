## script to download all models.
set -e
set -x

mkdir -p models/bpe
mkdir -p models/decoder
mkdir -p models/reranker_weights
mkdir -p models/embed
mkdir -p models/ngramlm
mkdir -p models/dicts

models_url=https://tinyurl.com/yywzdapm
echo "downloading pre-trained decoder model..."
model_file=decoder/lm.pt
curl -L -o models/$model_file $models_url/$model_file

echo "downloading models from Chollampatt and Ng, AAAI 2018"

models_url=https://tinyurl.com/yd6wvhgw/mlconvgec2018/models

echo "downloading embeddings..."
curl -L -o models/dicts/dict.src.txt $models_url/data_bin/dict.src.txt
curl -L -o models/dicts/dict.ctx.txt $models_url/data_bin/dict.src.txt
curl -L -o models/dicts/dict.trg.txt $models_url/data_bin/dict.trg.txt

echo "downloading BPE model..."
model_file=bpe/mlconvgec_aaai18_bpe.model
curl -L -o models/$model_file  $models_url/bpe_model/train.bpe.model

echo "downloading large n-gram LM (>150 GB)"
model_file=ngramlm/cclm.trie
curl -L -o models/$model_file https://tinyurl.com/yd6wvhgw/mlconvgec2018/models/lm/94Bcclm.trie