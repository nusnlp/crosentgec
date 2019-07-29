## script to download all models.
set -e
set -x

mkdir -p models/crosent
models_url=https://tinyurl.com/yywzdapm

for i in 1 2 3 4 ; do
    echo "downloadng crosent (model $i)"
    mkdir -p models/crosent/model$i
    curl -L -o models/crosent/model$i/checkpoint_best.pt $models_url/crosent/model$i/checkpoint_best.pt
done

echo "downloading reranker weights"
model_file=reranker_weights/weights.nucle-dev.txt
curl -L -o models/$model_file $models_url/$model_file