# Solving probability questions with neural networks 
This repository currently contains a pytorch implementation of LSTM and two different decoders, a discrete (n-bin) classifier and a regression component.

We use the data developed in the [nlp4plp project](https://dtai.cs.kuleuven.be/problog/natural_language).
The tasks are from the [bAbI](http://arxiv.org/abs/1502.05698) dataset.

## Requirements
- Python 3.6 or higher
- tested on numpy 1.15.4
- tested on scikit-learn 0.20.1
- tested on pytorch 0.4.1.post2


## Usage
Train and Evaluate model
```
python main.py --train 1 --lr 0.001 --hops 3 --eval 1 --saved-model-dir ./saved/ --data-dir ./data/tasks_1-20_v1-2/en-10k --task-number 1
```
