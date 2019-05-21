import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error

from corpus_util import Nlp4plpCorpus, Nlp4plpEncoder, Nlp4plpRegressionEncoder, Nlp4plpPointerNetEncoder
from net import LSTMClassifier, LSTMRegression, PointerNet


def get_correct_problems(test_corp, y_pred, bin_edges):
    correct = set()
    for inst, pred in zip(test_corp.insts, y_pred):
        if inst.ans_discrete == pred:
            correct.add(f"bin: {tuple(bin_edges[pred:pred+2])}\nid: {inst.id}\n{' '.join(inst.txt)}\n")
    return correct


def main():
    arg_parser = argparse.ArgumentParser(description="parser for End-to-End Memory Networks")
    arg_parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    arg_parser.add_argument("--bidir", action="store_true")
    arg_parser.add_argument("--cuda", type=int, default=0, help="train on GPU, default: 0")
    arg_parser.add_argument("--data-dir", type=str, default="",
                            help="path to folder from where data is loaded. Subfolder should be train/dev/test")
    arg_parser.add_argument("--dropout", type=float, default=0.0)
    arg_parser.add_argument("--embed-size", type=int, default=50, help="embedding dimension")
    arg_parser.add_argument("--epochs", type=int, default=1, help="number of training epochs, default: 100")
    arg_parser.add_argument("--hidden-dim", type=int, default=50, help="")
    # arg_parser.add_argument("--load-model-path", type=str, help="File path for the model.")
    arg_parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default: 0.01")
    # arg_parser.add_argument("--max-vocab-size", type=int, help="maximum number of words to keep, the rest is mapped to _UNK_", default=50000)
    arg_parser.add_argument("--model", type=str, help="lstm-enc-discrete-dec | lstm-enc-regression-dec | lstm-enc-pointer-dec")
    arg_parser.add_argument("--n-bins", type=int, default=10, help="number of bins for discretization of answers")
    arg_parser.add_argument("--n-layers", type=int, default=1, help="number of layers for the RNN")
    arg_parser.add_argument("--n-runs", type=int, default=5, help="number of runs to average over the results")
    arg_parser.add_argument("--pretrained-emb-path", type=str,
                            help="path to the txt file with word embeddings")
    arg_parser.add_argument("--print-correct", action="store_true")
    arg_parser.add_argument("--save-model", action="store_true")
    # arg_parser.add_argument("--test", type=int, default=1)
    # arg_parser.add_argument("--train", type=int, default=1)
    args = arg_parser.parse_args()

    # initialize corpora
    if args.model == "lstm-enc-discrete-dec":
        train_corp = Nlp4plpCorpus(args.data_dir + "train")
        dev_corp = Nlp4plpCorpus(args.data_dir + "dev")
        test_corp = Nlp4plpCorpus(args.data_dir + "test")

        train_corp.discretize(n_bins=args.n_bins)
        dev_corp.discretize(fitted_discretizer=train_corp.fitted_discretizer)
        test_corp.discretize(fitted_discretizer=train_corp.fitted_discretizer)

        label_size = len({inst.ans_discrete for inst in train_corp.insts})
    elif args.model == "lstm-enc-regression-dec":
        train_corp = Nlp4plpCorpus(args.data_dir + "train")
        dev_corp = Nlp4plpCorpus(args.data_dir + "dev")
        test_corp = Nlp4plpCorpus(args.data_dir + "test")
    elif args.model == "lstm-enc-pointer-dec":
        train_corp = Nlp4plpCorpus(args.data_dir + "train")
        dev_corp = Nlp4plpCorpus(args.data_dir + "dev")
        test_corp = Nlp4plpCorpus(args.data_dir + "test")

        train_corp.get_pointer_labels(label_type="group")
        dev_corp.get_pointer_labels(label_type="group")
        test_corp.get_pointer_labels(label_type="group")

        train_corp.remove_none_labels()
        dev_corp.remove_none_labels()
        test_corp.remove_none_labels()
    else:
        raise ValueError("Model should be 'lstm-enc-discrete-dec | lstm-enc-regression-dec | lstm-enc-pointer-dec'")

    test_score_runs = []
    for n in range(args.n_runs):
        if args.model == "lstm-enc-discrete-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpEncoder.from_corpus(train_corp, dev_corp)
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'label_size': label_size,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path
                          }
            classifier = LSTMClassifier(**net_params)
            eval_score = accuracy_score
        elif args.model == "lstm-enc-regression-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpRegressionEncoder.from_corpus(train_corp, dev_corp)
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path
                          }
            classifier = LSTMRegression(**net_params)
            eval_score = mean_absolute_error
        elif args.model == "lstm-enc-pointer-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpPointerNetEncoder.from_corpus(train_corp, dev_corp)
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'output_len': 1,  # decoder output length
                          'bidir': args.bidir
                          }
            classifier = PointerNet(**net_params)
            eval_score = accuracy_score
        else:
            raise ValueError("Model should be 'lstm-enc-discrete-dec | lstm-enc-regression-dec' | lstm-enc-pointer-dec")

        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

        classifier.train_model(train_corp, dev_corp, corpus_encoder, args.epochs, optimizer)

        # load model
        if args.model == 'lstm-enc-discrete-dec':
            classifier = LSTMClassifier.load(f_model='lstm_classifier.tar')
        elif args.model == 'lstm-enc-regression-dec':
            classifier = LSTMRegression.load(f_model='lstm_regression.tar')
        elif args.model == 'lstm-enc-pointer-dec':
            classifier = PointerNet.load(f_model='lstm_pointer.tar')
        else:
            raise ValueError("Model should be 'lstm-enc-discrete-dec | lstm-enc-regression-dec | lstm-enc-pointer-dec'")

        # get predictions
        y_pred, y_true = classifier.predict(test_corp, corpus_encoder)

        # compute scoring metrics
        test_acc = eval_score(y_true=y_true, y_pred=y_pred)
        if not args.print_correct and args.model == "lstm-enc-discrete-dec":
            correct = get_correct_problems(test_corp, y_pred, test_corp.fitted_discretizer.bin_edges_[0])
            print(correct)
        print('TEST SCORE: %.3f' % test_acc)
        test_score_runs.append(test_acc)
    print('AVG TEST SCORE over %d runs: %.3f' % (args.n_runs, np.mean(test_score_runs)))


if __name__ == '__main__':
    main()
