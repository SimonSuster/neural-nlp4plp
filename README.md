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