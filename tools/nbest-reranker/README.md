# N-best Reranker

Re-ranking N-best lists (MOSES format) using features like language models, edit operations etc. It is also easy to implement custom features.

Currently, tuning with BLEU and M2Scorer with MERT are supported

## Running the re-ranker

1. First augment the new feature using augment.py script

2. Then train the re-ranker using train.py script

3. Then rerank using rerank.py script
