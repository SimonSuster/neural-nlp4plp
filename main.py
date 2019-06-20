import argparse
import os
from datetime import datetime
import random

from pycocoevalcap.eval import COCOEvalCap
from util import f1_score

random.seed(0)

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)

from sklearn.metrics import accuracy_score, mean_absolute_error

from corpus_util import Nlp4plpCorpus, Nlp4plpEncoder, Nlp4plpRegressionEncoder, Nlp4plpPointerNetEncoder, \
    Nlp4plpEncDecEncoder
from net import LSTMClassifier, LSTMRegression, PointerNet, EncoderDecoder


def get_correct_problems(test_corp, y_pred, bin_edges):
    correct = set()
    for inst, pred in zip(test_corp.insts, y_pred):
        if inst.ans_discrete == pred:
            correct.add(f"bin: {tuple(bin_edges[pred:pred + 2])}\nid: {inst.id}\n{' '.join(inst.txt)}\n")
    return correct


def inspect_encdec(corp, label_vocab, _y_true, _y_pred):
    print("Inspecting predictions from the best model:")
    correct = []
    incorrect = []
    for c, (y_t, y_p) in enumerate(zip(_y_true, _y_pred)):
        y_t = list(y_t)
        y_p = list(y_p)
        if y_t == y_p:
            if len(correct) == 30:
                continue
            correct.append((corp.insts[c].f, y_t, y_p))
        else:
            if len(incorrect) == 30:
                continue
            incorrect.append((corp.insts[c].f, y_t, y_p))
        if len(correct) == 30 and len(incorrect) == 30:
            break

    print("CORRECT:")
    for f, y_t, y_p in correct:
        print("\n"+f)
        print([label_vocab.idx2word[y] for y in y_t])
        print([label_vocab.idx2word[y] for y in y_p])
    print("INCORRECT:")
    for f, y_t, y_p in incorrect:
        print("\n" + f)
        print([label_vocab.idx2word[y] for y in y_t])
        print([label_vocab.idx2word[y] for y in y_p])


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
    arg_parser.add_argument("--feat_embed-size", type=int, default=10, help="embedding dimension for external features")
    arg_parser.add_argument("--hidden-dim", type=int, default=50, help="")
    arg_parser.add_argument("--inspect", action="store_true")
    # arg_parser.add_argument("--load-model-path", type=str, help="File path for the model.")
    arg_parser.add_argument("--label-type", type=str,
                            help="group | take | take_wr | both_take | take3 | take_declen2 | take_wr_declen2 | take_declen3 | take_wr_declen3 | both_take_declen3. To use with PointerNet.")
    arg_parser.add_argument("--label-type-dec", type=str,
                            help="predicates | predicates-all | predicates-arguments-all. To use with EncDec.")
    arg_parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default: 0.01")
    # arg_parser.add_argument("--max-vocab-size", type=int, help="maximum number of words to keep, the rest is mapped to _UNK_", default=50000)
    arg_parser.add_argument("--max-output-len", type=int, default=50,
                            help="Maximum decoding length for EncDec models at prediction time.")
    model_names = "lstm-enc-discrete-dec | lstm-enc-regression-dec | lstm-enc-pointer-dec | lstm-enc-dec"
    arg_parser.add_argument("--model", type=str, help=f"{model_names}")
    arg_parser.add_argument("--n-bins", type=int, default=10, help="number of bins for discretization of answers")
    arg_parser.add_argument("--n-layers", type=int, default=1, help="number of layers for the RNN")
    arg_parser.add_argument("--n-runs", type=int, default=5, help="number of runs to average over the results")
    arg_parser.add_argument("--pretrained-emb-path", type=str,
                            help="path to the txt file with word embeddings")
    arg_parser.add_argument("--feat-pos", action="store_true", help="use PoS features in the encoder")
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

        train_corp.get_pointer_labels(label_type=args.label_type)
        dev_corp.get_pointer_labels(label_type=args.label_type)
        test_corp.get_pointer_labels(label_type=args.label_type)

        train_corp.remove_none_labels()
        dev_corp.remove_none_labels()
        test_corp.remove_none_labels()
    elif args.model == "lstm-enc-dec":
        train_corp = Nlp4plpCorpus(args.data_dir + "train")
        dev_corp = Nlp4plpCorpus(args.data_dir + "dev")
        test_corp = Nlp4plpCorpus(args.data_dir + "test")

        train_corp.get_labels(label_type=args.label_type_dec, max_output_len=args.max_output_len)
        dev_corp.get_labels(label_type=args.label_type_dec, max_output_len=args.max_output_len)
        test_corp.get_labels(label_type=args.label_type_dec)

        train_corp.remove_none_labels()
        dev_corp.remove_none_labels()
        test_corp.remove_none_labels()
    else:
        raise ValueError(f"Model should be '{model_names}'")

    feature_encoder = None
    test_score_runs = []
    if args.model == "lstm-enc-dec":
        test_score_f1_runs = []
        test_score_bleu4_runs = []
    for n in range(args.n_runs):
        if args.model == "lstm-enc-discrete-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpEncoder.from_corpus(train_corp, dev_corp)
            f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
            print(f"f_model: {f_model}")
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'label_size': label_size,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'f_model': f_model
                          }
            classifier = LSTMClassifier(**net_params)
            eval_score = accuracy_score
        elif args.model == "lstm-enc-regression-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpRegressionEncoder.from_corpus(train_corp, dev_corp)
            f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
            print(f"f_model: {f_model}")
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'f_model': f_model
                          }
            classifier = LSTMRegression(**net_params)
            eval_score = mean_absolute_error
        elif args.model == "lstm-enc-pointer-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpPointerNetEncoder.from_corpus(train_corp, dev_corp)
            f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
            print(f"f_model: {f_model}")
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'output_len': int(args.label_type[-1]) if "declen" in args.label_type else 1,
                          # decoder output length
                          'f_model': f_model,
                          'bidir': args.bidir
                          }
            if args.feat_pos:
                feature_encoder = Nlp4plpPointerNetEncoder.feature_from_corpus(train_corp, dev_corp, feat_type=["pos"])
                net_params['feature_idx'] = feature_encoder.vocab.word2idx
                net_params['feat_size'] = feature_encoder.vocab.size
                net_params['feat_padding_idx'] = feature_encoder.vocab.pad
                net_params['feat_emb_dim'] = args.feat_embed_size

            classifier = PointerNet(**net_params)
            eval_score = accuracy_score
        elif args.model == "lstm-enc-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpEncDecEncoder.from_corpus(train_corp, dev_corp)
            # print(corpus_encoder.label_vocab.to_dict()["word2idx"])
            print(f"n labels: {len(corpus_encoder.label_vocab)}")
            # max_output_len = max([len(inst.label) for inst in train_corp.insts + dev_corp.insts])
            f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
            print(f"f_model: {f_model}")
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'label_padding_idx': corpus_encoder.label_vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'label_idx': corpus_encoder.label_vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'max_output_len': args.max_output_len,
                          'label_size': len(corpus_encoder.label_vocab),
                          'f_model': f_model,
                          'bidir': args.bidir
                          }
            if args.feat_pos:
                feature_encoder = Nlp4plpEncoder.feature_from_corpus(train_corp, dev_corp, feat_type=["pos"])
                net_params['feature_idx'] = feature_encoder.vocab.word2idx
                net_params['feat_size'] = feature_encoder.vocab.size
                net_params['feat_padding_idx'] = feature_encoder.vocab.pad
                net_params['feat_emb_dim'] = args.feat_embed_size

            classifier = EncoderDecoder(**net_params)
            eval_score = accuracy_score

        else:
            raise ValueError(f"Model should be '{model_names}'")

        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

        print(os.path.abspath('../out/' + classifier.f_model))
        classifier.train_model(train_corp, dev_corp, corpus_encoder, feature_encoder, args.epochs, optimizer)

        # load model
        if args.model == 'lstm-enc-discrete-dec':
            classifier = LSTMClassifier.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-regression-dec':
            classifier = LSTMRegression.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-pointer-dec':
            classifier = PointerNet.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-dec':
            classifier = EncoderDecoder.load(f_model=net_params['f_model'])
        else:
            raise ValueError(f"Model should be '{model_names}'")

        # get predictions
        _y_pred, _y_true = classifier.predict(test_corp, feature_encoder, corpus_encoder)

        # for accuracy calculation
        if args.model == "lstm-enc-dec" or net_params["output_len"] > 1:
            y_true = [str(y) for y in _y_true]
            y_pred = [str(y) for y in _y_pred]
        else:
            y_true = _y_true
            y_pred = _y_pred

        # compute scoring metrics
        test_acc = eval_score(y_true=y_true, y_pred=y_pred)

        if not args.print_correct and args.model == "lstm-enc-discrete-dec":
            correct = get_correct_problems(test_corp, y_pred, test_corp.fitted_discretizer.bin_edges_[0])
            print(correct)
        if args.model == "lstm-enc-dec":
            test_f1 = np.mean([f1_score(y_true=t, y_pred=p) for t, p in zip(_y_true, _y_pred)])
            y_true_dict = {c: [" ".join([str(i) for i in t])] for c, t in enumerate(_y_true)}
            y_pred_dict = {c: [" ".join([str(i) for i in p])] for c, p in enumerate(_y_pred)}
            test_coco_eval = COCOEvalCap(y_true_dict, y_pred_dict)
            test_coco_eval.evaluate()
            test_coco = test_coco_eval.eval
            test_bleu4 = test_coco["Bleu_4"]
            print('TEST SCORE: acc: %.3f, f1: %.3f, bleu4: %.3f' % (test_acc, test_f1, test_bleu4))
            test_score_runs.append(test_acc)
            test_score_f1_runs.append(test_f1)
            test_score_bleu4_runs.append(test_bleu4)
        else:
            print('TEST SCORE: %.3f' % test_acc)
            test_score_runs.append(test_acc)
    if args.model == "lstm-enc-dec":
        print('AVG TEST SCORE over %d runs: %.3f, f1: %.3f, bleu4: %.3f' % (
            args.n_runs, np.mean(test_score_runs), np.mean(test_score_f1_runs), np.mean(test_score_bleu4_runs)))
    else:
        print('AVG TEST SCORE over %d runs: %.3f' % (args.n_runs, np.mean(test_score_runs)))

    if args.inspect:
        #inspect(test_corp, _y_true, _y_pred)
        # get dev predictions
        _y_pred, _y_true = classifier.predict(dev_corp, feature_encoder, corpus_encoder)
        inspect_encdec(dev_corp, corpus_encoder.label_vocab, _y_true, _y_pred)

    if not args.save_model:
        classifier.remove(f_model=classifier.f_model)


if __name__ == '__main__':
    main()
