# Solving probability questions with neural networks 
This repository currently contains a pytorch implementation of LSTM and two different decoders, a discrete (n-bin) classifier and a regression component.

We use the data developed in the [nlp4plp project](https://dtai.cs.kuleuven.be/problog/natural_language).

## Requirements
- Python 3.6 or higher
- tested on numpy 1.15.4
- tested on scikit-learn 0.20.1
- tested on pytorch 0.4.1.post2


## Usage
Train and Evaluate model
```
python3.7 main.py --data-dir /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/ --model lstm-enc-discrete-dec --pretrained-emb-path /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/news_embs/embs/1/embeddings --n-bins 20 --epochs 30 --n-runs 5
```
