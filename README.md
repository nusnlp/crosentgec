Cross-sentence Grammatical Error Correction
-------------------------------------------

This repository contains the code and models to train and test cross-sentence grammatical error correction models using convolutional sequence-to-sequence models.

If you use this code, please cite this paper:
```
@InProceedings{chollampatt2019crosent,
  author    = {Shamil Chollampatt and Weiqi Wang and Hwee Tou Ng},
  title     = {Cross-Sentence Grammatical Error Correction},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year      = {2019}
}
```

## Prerequisites

Data Processing:
* Python 2.7
* To generate exactly the excat same data with same tokenization, you may require NLTK v2.0b7 and LangID.py v1.1.6.

Training Baseline and CroSent models:
* Python 3.6
* PyTorch 0.4.1

Training, and running rescorer:
* Python 2.7
* Moses v3

For training NUS3 models:
  - Fast_align, Moses: for computing word alignments for edit weighted log likelihood loss

## Decoding using pre-trained cross-sentence GEC models

1. Run `prepare_test.sh` to prepare the test datasets.

2. Download all pre-requiste components (BPE model, embeddings, and pre-trained decoder)  using the `download.sh`

3. Download CroSent models and dictionaries using `download_pretrained_crosent.sh` script.

4. Decode development/test sets with `decode.sh`.

```
./decode.sh $testset $modelpath $dictdir $optionalgpu
```
`$testset` is the test dataset name. The test dataset files are in the format `data/test/$testset/$testset.tok.src` (for the input source sentences) and `data/test/$testset/$testset.tok.ctx` (for the context sentences, i.e. 2 previous sentences per line)

`$modelpath`: could be a file for decoding using a single model or a directory for ensemble (any model with the name checkpoint_best.pt within the specified directory will be used in the ensemble). If single model, the decoder will output the files into a directory in the same location as the model path, with the name same as the model path with a prefix `outputs.`. If ensemble, the decoder will output the files into `outptus/` directory within $model_path

`$dictdir` contains the path to the dictionaries. For pre-trained models it is `models/dicts`

`$optionalgpu` is an optional parameter indicating GPU id to run the decoding on (default=0).

5. Run rearnker using the downloaded weights:
```
./reranker_run.sh $outputsdir $testset $weightsfile $optionalgpu
```
where `$outputsdir` is the directory which contains the output of the decoding and `$weightsfile` is the paths to trained weights (in the case of pretrained weights, it is `models/reranker_weights/weights.nucle_dev.txt`)

## Training from scratch

### Data preparation

Download the required datasets and run `prepare_data.sh` with the paths to Lang-8 and NUCLE to prepare the datasets.

### Training

Download all pre-requiste components (BPE model, dictionary files, embeddings, and pre-trained decoder)  using the `download.sh`

Each training script `train_*.sh` has a parameter to specify the random seed value. To train 4 different models, run the training script 4 times by variying the seed values (e.g., 1, 2, 3, 4)

For training the baseline models use `train_baseline.sh` script.

For training the crosent models, use `train_crosent.sh` script.

For training the NUS2 model, use `train_nus2.sh` script.

For training the NUS3 model
1. Generate alignments using fastalign (Requires `fast_align` and `moses` under `tools/` directory), run `create_alignment.py data/processed`
2. Run `train_nus3.sh` script.

For training the reranker:

1. Decode development dataset using `./decode.sh` (steps mentioned earlier). Set `$outputsdir` to the output directory of this decoding step.

2. Run `./reranker_train.sh $outputsdir $devset $optionalgpu`


## License

The source code is licensed under GNU GPL 3.0 (see [LICENSE](LICENSE.md)) for non-commerical use. For commercial use of this code, separate commercial licensing is also available. Please contact Prof. Hwee Tou Ng (nght@comp.nus.edu.sg)
