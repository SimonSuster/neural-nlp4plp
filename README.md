# Solving probability questions with neural networks 
This repository currently contains a pytorch implementation of an LSTM encoder with different types of decoders for solving probability word problems.
The LSTM encoder uses pretrained word embeddings and embedded PoS features.
The decoders depend on the type of prediction:
- a multiclass predictor for end2end classification into N bins. Here, the probability space is discretized.
- a regressor for end2end prediction of probability. We a simple single node layer on top (with or without sigmoid) to support continuous output. The loss is mean squared error.
- a pointer decoder, which predicts for a preselected predicate its arguments. The implementation is based on the idea of pointer networks.
- a sequence-of-predicates decoder

We use the data developed in the [nlp4plp project](https://dtai.cs.kuleuven.be/problog/natural_language).

## Requirements
- Python 3.6 or higher
- tested on numpy 1.15.4
- tested on scikit-learn 0.20.1
- tested on pytorch 0.4.1.post2
- comet-ml

Obtain the Comet-ml key by logging in on their website, and selecting a new project. Then:
```
export COMET_API_KEY="type-your-key"
```
## Usage
Train and evaluate the model

LSTM classifier with discretization into n bins:

```
python3.7 main.py --data-dir /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/ --model lstm-enc-discrete-dec --pretrained-emb-path /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/news_embs/embs/1/embeddings --n-bins 20 --epochs 30 --n-runs 5
```

LSTM regressor:

```
python3.7 main.py --data-dir /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/ --model lstm-enc-regression-dec --pretrained-emb-path /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/news_embs/embs/1/embeddings --epochs 30 --n-runs 5
```

Others TODO.

### Re-ranking
To train a re-ranker:
```
PYTHONHASHSEED=0 python3.6 main_reranker.py --data-dir resources/examples_splits_nums_mapped/ --embed-size 100 --ret-period 30 --feat-type pos rels num --beam-decoding --beam-width 5 --beam-topk 5 --model-path 20191118_151545_853074  --rank-discrete-feat-type score rank --lr 0.0001
```

This will obtain the predictions from a pre-trained base model (`-model-path`), then train a re-ranker based on those predictions. The prediction step uses a beam size of 5 and also keeps the 5 topmost predictions. The training of the re-ranking model simply consists of predicting for each prediction whether it is correct or not (independently of all others).   