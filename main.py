import argparse
import os
import pickle
import random
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime

from pycocoevalcap.eval import COCOEvalCap
from util import f1_score, FileUtils

random.seed(1)

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)

from sklearn.metrics import accuracy_score, mean_absolute_error

from corpus_util import Nlp4plpCorpus, Nlp4plpEncoder, Nlp4plpRegressionEncoder, Nlp4plpPointerNetEncoder, \
    Nlp4plpEncDecEncoder, get_bert_embs, Nlp4plpEncSplitDecEncoder
from net import LSTMClassifier, LSTMRegression, PointerNet, EncoderDecoder, EncoderSplitDecoder


def get_correct_problems(test_corp, y_pred, bin_edges):
    correct = set()
    for inst, pred in zip(test_corp.insts, y_pred):
        if inst.ans_discrete == pred:
            correct.add(f"bin: {tuple(bin_edges[pred:pred + 2])}\nid: {inst.id}\n{' '.join(inst.txt)}\n")
    return correct


def save_preds_encdec(corp, label_vocab, _y_true, _y_pred, f_name, dir_out="../out/"):
    print("Saving predictions from the best model:")
    f_name = f"{f_name}.json"
    all = {}
    for c, (y_t, y_p) in enumerate(zip(_y_true, _y_pred)):
        y_t = list(y_t)
        y_p = list(y_p)
        if y_t == y_p:
            correct = True
        else:
            correct = False
        t = [label_vocab.idx2word[y] for y in y_t]
        p = [label_vocab.idx2word[y] for y in y_p]
        all[corp.insts[c].f] = {"true": t, "pred": p, "correct": correct}

    FileUtils.write_json(all, f_name, dir_out)
    print(f"Writing predictions to {dir_out}{f_name}")


def final_repl(n, num2n_map):
    k = []
    for i in n:
        if i.startswith(")"):
            k.append(")")
        else:
            if num2n_map is not None:
                inv_num2n_map = {v: k for k, v in num2n_map.items()}
                if i in inv_num2n_map:
                    k.append(inv_num2n_map[i])
                    continue
            k.append(i)
    new_k = " ".join(k).replace(" ", "").replace(").", ").\n")
    return new_k


def save_preds_encdec_pl(corp, _y_true, _y_pred, log_name, label_vocab=None, dir_out="../out/"):
    """ as a prolog program """
    print("Saving predictions from the best model:")
    dir_out = f"{dir_out}log_w{log_name}/"
    print(f"Save preds dir: {dir_out}")
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for c, (y_t, y_p) in enumerate(zip(_y_true, _y_pred)):
        f_name_t = os.path.basename(corp.insts[c].f + "_t")
        f_name_p = os.path.basename(corp.insts[c].f + "_p")
        with open(dir_out + f_name_t, "w") as f_out_t, open(dir_out + f_name_p, "w") as f_out_p:
            y_t = list(y_t)
            y_p = list(y_p)
            if label_vocab is not None:
                t = [label_vocab.idx2word[y] for y in y_t]
                p = [label_vocab.idx2word[y] for y in y_p]
            else:
                t = y_t
                p = y_p
            t = final_repl(t, corp.insts[c].num2n_map)
            p = final_repl(p, corp.insts[c].num2n_map)
            f_out_t.write(t)
            f_out_p.write(p)


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
        print("\n" + f)
        print([label_vocab.idx2word[y] for y in y_t])
        print([label_vocab.idx2word[y] for y in y_p])
    print("INCORRECT:")
    for f, y_t, y_p in incorrect:
        print("\n" + f)
        print([label_vocab.idx2word[y] for y in y_t])
        print([label_vocab.idx2word[y] for y in y_p])


def augment_train(train_corp, fac=10):
    fs_new, insts_new = [], []
    for f, inst in zip(train_corp.fs, train_corp.insts):
        fs_new.extend([f] * fac)
        for i in range(fac):
            inst_new = deepcopy(inst)
            np.random.shuffle(inst_new.statements)
            insts_new.append(inst_new)
    train_corp.fs = train_corp.fs + fs_new
    train_corp.insts = train_corp.insts + insts_new

    return train_corp


def main():
    arg_parser = argparse.ArgumentParser(description="parser for End-to-End Memory Networks")
    arg_parser.add_argument("--attention-plot", action="store_true")
    arg_parser.add_argument("--augment-train", action="store_true")
    arg_parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    arg_parser.add_argument("--beam-decoding", action="store_true")
    arg_parser.add_argument("--bert", action="store_true",
                            help="use bert to initialize token embeddings")
    arg_parser.add_argument("--no-load-bert", action="store_true",
                            help="will not load presaved embs, but get them from scratch")
    arg_parser.add_argument("--bert-tok-emb-path", type=str, default="",
                            help="path to bert embeddings for all toks in train/dev/test in json format")
    arg_parser.add_argument("--bidir", action="store_true")
    arg_parser.add_argument("--constrained-decoding", nargs='+',
                            help="List of modifications to use:\n/"
                                 "**mod1**: label_dec1 is embedded using label_embeddings, label_dec1 as input to LSTM_dec2;\n/"
                                 "**mod2**: label_dec1 is represented as output distribution over all labels of dec1, label_dec1 as input to output layer of dec2\n/"
                                 "**mod3**: masking for types (numbers) on output of dec2\n/"
                                 "**mod4**: mask to 0 all outputs for nsymbs where n > max_n in num2n dict. Applies to single dec only.\n/"
                                 "**mod5**: parent feeding (in out)\n/"
                                 "**mod6**: parent feeding (in attention)")
    arg_parser.add_argument("--convert-consts", type=str, help="conv | our-map | no-our-map | no. \n/"
                                                                              "conv-> txt: -; stats: num_sym+ent_sym.\n/"
                                                                              "our-map-> txt: num_sym; stats: num_sym(from map)+ent_sym;\n/"
                                                                              "no-our-map-> txt: -; stats: num_sym(from map)+ent_sym;\n/"
                                                                              "no-> txt: -; stats: -, only ent_sym;\n/"
                                                                              "no-ent-> txt: -; stats: -, no ent_sym;\n/")
    arg_parser.add_argument("--cuda", type=int, default=0, help="train on GPU, default: 0")
    arg_parser.add_argument("--data-dir", type=str, default="",
                            help="path to folder from where data is loaded. Subfolder should be train/dev/test")
    arg_parser.add_argument("--debug", action="store_true")
    arg_parser.add_argument("--dropout", type=float, default=0.0)
    arg_parser.add_argument("--embed-size", type=int, default=50, help="embedding dimension")
    arg_parser.add_argument("--epochs", type=int, default=1, help="number of training epochs, default: 100")
    arg_parser.add_argument("--feat-onehot", action="store_true",
                            help="use onehot feature encoding instead of embedded")
    arg_parser.add_argument('--feat-type', nargs='+', help="Which feature to use: pos | rels | num | sen_ns")
    arg_parser.add_argument("--feat_embed-size", type=int, default=10, help="embedding dimension for external features")
    arg_parser.add_argument("--hidden-dim", type=int, default=50, help="")
    arg_parser.add_argument("--inspect", action="store_true")
    # arg_parser.add_argument("--load-model-path", type=str, help="File path for the model.")
    arg_parser.add_argument("--label-type", type=str,
                            help="group | take | take_wr | both_take | take3 | take_declen2 | take_wr_declen2 | take_declen3 | take_wr_declen3 | both_take_declen3. To use with PointerNet.")
    arg_parser.add_argument("--label-type-dec", type=str,
                            help="predicates | n-predicates | n-full | predicates-all | predicates-arguments-all | full-pl | full-pl-no-arg-id | full-pl-split | full-pl-split-plc | full-pl-split-stat-dyn. To use with EncDec.")
    arg_parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default: 0.001")
    # arg_parser.add_argument("--max-vocab-size", type=int, help="maximum number of words to keep, the rest is mapped to _UNK_", default=50000)
    arg_parser.add_argument("--max-output-len", type=int, default=500,
                            help="Maximum decoding length for EncDec models at prediction time.")
    model_names = "lstm-enc-discrete-dec | lstm-enc-regression-dec | lstm-enc-pointer-dec | lstm-enc-dec | lstm-enc-split-dec"
    arg_parser.add_argument("--model", type=str, help=f"{model_names}")
    arg_parser.add_argument("--n-bins", type=int, default=10, help="number of bins for discretization of answers")
    arg_parser.add_argument("--n-layers", type=int, default=1, help="number of layers for the RNN")
    arg_parser.add_argument("--n-runs", type=int, default=5, help="number of runs to average over the results")
    arg_parser.add_argument("--oracle-dec1", action="store_true",
                            help="use gold seq instead of dec1 output to feed to dec2")
    arg_parser.add_argument("--pretrained-emb-path", type=str,
                            help="path to the txt file with word embeddings")
    arg_parser.add_argument("--print-correct", action="store_true")
    arg_parser.add_argument("--ret-period", type=int, default=20, help="stop training if no improvement in both acc and f1 during last 'ret-period' epochs")
    arg_parser.add_argument("--save-model", action="store_true")
    # arg_parser.add_argument("--test", type=int, default=1)
    # arg_parser.add_argument("--train", type=int, default=1)
    args = arg_parser.parse_args()
    if args.bert:
        args.embed_size = 1024
    if args.convert_consts in {"conv"}:
        assert "nums_mapped" not in args.data_dir
    elif args.convert_consts in {"our-map", "no-our-map", "no", "no-ent"}:
        assert "nums_mapped" in args.data_dir
    else:
        if args.convert_consts is not None:
            raise ValueError

    print(args)
    # initialize corpora
    if args.model == "lstm-enc-discrete-dec" and args.label_type_dec not in {"n-predicates", "n-full"}:
        train_corp = Nlp4plpCorpus(args.data_dir + "train")
        dev_corp = Nlp4plpCorpus(args.data_dir + "dev")
        test_corp = Nlp4plpCorpus(args.data_dir + "test")

        train_corp.discretize(n_bins=args.n_bins)
        dev_corp.discretize(fitted_discretizer=train_corp.fitted_discretizer)
        test_corp.discretize(fitted_discretizer=train_corp.fitted_discretizer)
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
    elif args.model in {"lstm-enc-dec", "lstm-enc-split-dec", "lstm-enc-discrete-dec"}:
        train_corp = Nlp4plpCorpus(args.data_dir + "train", args.convert_consts)
        if args.augment_train:
            train_corp = augment_train(train_corp)
        print(f"Size of train: {len(train_corp.insts)}")
        dev_corp = Nlp4plpCorpus(args.data_dir + "dev", args.convert_consts)
        test_corp = Nlp4plpCorpus(args.data_dir + "test", args.convert_consts)
        if args.debug:
            train_corp.fs = train_corp.fs[:10]
            train_corp.insts = train_corp.insts[:10]
            dev_corp.fs = dev_corp.fs[:10]
            dev_corp.insts = dev_corp.insts[:10]
            test_corp.insts = test_corp.insts[:10]

        train_corp.get_labels(label_type=args.label_type_dec, max_output_len=args.max_output_len)
        dev_corp.get_labels(label_type=args.label_type_dec, max_output_len=args.max_output_len)
        test_corp.get_labels(label_type=args.label_type_dec)

        c = train_corp.add_tok_ids()
        c = dev_corp.add_tok_ids(c)
        c = test_corp.add_tok_ids(c)

        if args.label_type_dec not in {"n-predicates", "n-full"}:
            train_corp.remove_none_labels()
            dev_corp.remove_none_labels()
            test_corp.remove_none_labels()

        if args.bert:
            if args.no_load_bert:
                print("collecting bert embeddings...")
                from bert_serving.client import BertClient
                bc = BertClient()  # root@ssuster:/home/suster# bert-serving-start -pooling_strategy NONE -model_dir /nas/corpora/bert_models/uncased_L-24_H-1024_A-16/ -num_worker=1
                bert_embs = get_bert_embs(train_corp.insts, bc)
                bert_embs = get_bert_embs(dev_corp.insts, bc, bert_embs)
                bert_embs = get_bert_embs(test_corp.insts, bc, bert_embs)
                pickle.dump(bert_embs, open(args.bert_tok_emb_path, "wb"))
                print("finished collecting bert embeddings")
            else:
                # bert_embs = load_json(args.bert_tok_emb_path+".json")
                bert_embs = pickle.load(open(args.bert_tok_emb_path, "rb"))
    else:
        raise ValueError(f"Model should be '{model_names}'")

    feature_encoder = None
    test_score_runs = []
    if args.model in {"lstm-enc-dec", "lstm-enc-split-dec"}:
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
                          'label_padding_idx': corpus_encoder.label_vocab.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'label_size': len(corpus_encoder.label_vocab),
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'f_model': f_model,
                          'cuda': args.cuda
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
                          'f_model': f_model,
                          'cuda': args.cuda

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
                          'bidir': args.bidir,
                          'cuda': args.cuda
                          }
            if args.feat_type:
                feature_encoder = Nlp4plpPointerNetEncoder.feature_from_corpus(train_corp, dev_corp,
                                                                               feat_type=args.feat_type)
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
            print(corpus_encoder.label_vocab.word2idx)
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
                      'bidir': args.bidir,
                      'bert_embs': bert_embs if args.bert else None,
                      'constrained_decoding': args.constrained_decoding,
                      'cuda': args.cuda
            }
            if args.feat_type:
                feature_encoder = Nlp4plpEncoder.feature_from_corpus(train_corp, dev_corp, feat_type=args.feat_type)
                net_params['feature_idx'] = feature_encoder.vocab.word2idx
                net_params['feat_size'] = feature_encoder.vocab.size
                net_params['feat_padding_idx'] = feature_encoder.vocab.pad
                net_params['feat_emb_dim'] = args.feat_embed_size
                net_params['feat_type'] = args.feat_type
                net_params['feat_onehot'] = args.feat_onehot

            classifier = EncoderDecoder(**net_params)
            eval_score = accuracy_score
        elif args.model == "lstm-enc-split-dec":
            # initialize vocab
            corpus_encoder = Nlp4plpEncSplitDecEncoder.from_corpus(train_corp, dev_corp)
            # print(corpus_encoder.label_vocab.to_dict()["word2idx"])
            print(f"n labels: {len(corpus_encoder.label_vocab)}")
            print(f"n labels2: {len(corpus_encoder.label_vocab2)}")
            # max_output_len = max([len(inst.label) for inst in train_corp.insts + dev_corp.insts])
            f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
            print(f"f_model: {f_model}")
            net_params = {'n_layers': args.n_layers,
                          'hidden_dim': args.hidden_dim,
                          'vocab_size': corpus_encoder.vocab.size,
                          'padding_idx': corpus_encoder.vocab.pad,
                          'label_padding_idx': corpus_encoder.label_vocab.pad,
                          'label_padding_idx2': corpus_encoder.label_vocab2.pad,
                          'embedding_dim': args.embed_size,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'word_idx': corpus_encoder.vocab.word2idx,
                          'label_idx': corpus_encoder.label_vocab.word2idx,
                          'label_idx2': corpus_encoder.label_vocab2.word2idx,
                          'pretrained_emb_path': args.pretrained_emb_path,
                          'max_output_len': args.max_output_len,
                          'label_size': len(corpus_encoder.label_vocab),
                          'label_size2': len(corpus_encoder.label_vocab2),
                          'f_model': f_model,
                          'bidir': args.bidir,
                          'bert_embs': bert_embs if args.bert else None,
                          'oracle_dec1': args.oracle_dec1,
                          'constrained_decoding': args.constrained_decoding,
                          'label_type_dec': args.label_type_dec,
                          'cuda': args.cuda
                          }
            if args.feat_type:
                feature_encoder = Nlp4plpEncoder.feature_from_corpus(train_corp, dev_corp, feat_type=args.feat_type)
                net_params['feature_idx'] = feature_encoder.vocab.word2idx
                net_params['feat_size'] = feature_encoder.vocab.size
                net_params['feat_padding_idx'] = feature_encoder.vocab.pad
                net_params['feat_emb_dim'] = args.feat_embed_size
                net_params['feat_type'] = args.feat_type
                net_params['feat_onehot'] = args.feat_onehot

            classifier = EncoderSplitDecoder(**net_params)
            eval_score = accuracy_score

        else:
            raise ValueError(f"Model should be '{model_names}'")

        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

        print(os.path.abspath('../out/' + classifier.f_model))
        classifier.train_model(train_corp, dev_corp, corpus_encoder, feature_encoder, args.epochs, args.ret_period, optimizer)


        # load model
        if args.model == 'lstm-enc-discrete-dec':
            classifier = LSTMClassifier.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-regression-dec':
            classifier = LSTMRegression.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-pointer-dec':
            classifier = PointerNet.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-dec':
            classifier = EncoderDecoder.load(f_model=net_params['f_model'])
        elif args.model == 'lstm-enc-split-dec':
            classifier = EncoderSplitDecoder.load(f_model=net_params['f_model'])
        else:
            raise ValueError(f"Model should be '{model_names}'")

        # get predictions
        if args.model == "lstm-enc-split-dec":
            _y_pred, _y_true, (_y_pred1, _y_pred2, _y_true1, _y_true2) = classifier.predict(test_corp, feature_encoder,
                                                                                            corpus_encoder, args.attention_plot)
            y_pred1 = [str(y) for y in _y_pred1]
            y_pred2 = [str(y) for y in _y_pred2]
            y_true1 = [str(y) for y in _y_true1]
            y_true2 = [str(y) for y in _y_true2]
        else:
            _y_pred, _y_true = classifier.predict(test_corp, feature_encoder, corpus_encoder, args.attention_plot, args.beam_decoding)

        # for accuracy calculation
        if args.model in {"lstm-enc-dec", "lstm-enc-split-dec"}: # or net_params["output_len"] > 1:
            y_true = [str(y) for y in _y_true]
            if args.beam_decoding:
                # 1st best
                _y_pred = [preds[0] for preds in _y_pred]
            y_pred = [str(y) for y in _y_pred]
        else:
            y_true = _y_true
            y_pred = _y_pred

        # compute scoring metrics
        test_acc = eval_score(y_true=y_true, y_pred=y_pred)
        if args.oracle_dec1:
            test_acc1 = accuracy_score(y_true=y_true1, y_pred=y_pred1)
            test_acc2 = accuracy_score(y_true=y_true2, y_pred=y_pred2)
            print(f"acc dec1: {test_acc1}, acc dec2: {test_acc2}")
        if not args.print_correct and (args.model == "lstm-enc-discrete-dec" and args.label_type_dec not in {"n-predicates", "n-full"}):
            correct = get_correct_problems(test_corp, y_pred, test_corp.fitted_discretizer.bin_edges_[0])
            print(correct)
        if args.model in {"lstm-enc-dec", "lstm-enc-split-dec"}:
            test_f1 = np.mean([f1_score(y_true=t, y_pred=p) for t, p in zip(_y_true, _y_pred)])
            y_true_dict = {c: [" ".join([str(i) for i in t])] for c, t in enumerate(_y_true)}
            y_pred_dict = {c: [" ".join([str(i) for i in p])] for c, p in enumerate(_y_pred)}
            test_coco_eval = COCOEvalCap(y_true_dict, y_pred_dict)
            test_coco_eval.evaluate()
            test_coco = test_coco_eval.eval
            test_bleu4 = test_coco["Bleu_4"]
            print('TEST SCORE: acc: %.3f, f1: %.3f, bleu4: %.3f, total: %i' % (
            test_acc, test_f1, test_bleu4, len(y_pred)))
            test_score_runs.append(test_acc)
            test_score_f1_runs.append(test_f1)
            test_score_bleu4_runs.append(test_bleu4)
        else:
            print('TEST SCORE: %.3f' % test_acc)
            test_score_runs.append(test_acc)
    if args.model in {"lstm-enc-dec", "lstm-enc-split-dec"}:
        print('AVG TEST SCORE over %d runs: %.3f, f1: %.3f, bleu4: %.3f' % (
            args.n_runs, np.mean(test_score_runs), np.mean(test_score_f1_runs), np.mean(test_score_bleu4_runs)))
    else:
        print('AVG TEST SCORE over %d runs: %.3f' % (args.n_runs, np.mean(test_score_runs)))

    if args.inspect:
        # inspect(test_corp, _y_true, _y_pred)
        # get dev predictions
        #if args.model == "lstm-enc-split-dec":
        #    _y_pred, _y_true, (_y_pred1, _y_pred2, _y_true1, _y_true2) = classifier.predict(test_corp, feature_encoder,
        #                                                                                    corpus_encoder)
        #else:
        #    _y_pred, _y_true = classifier.predict(test_corp, feature_encoder, corpus_encoder)
        if args.save_model:
            if args.label_type_dec == "full-pl":
                save_preds_encdec_pl(test_corp, _y_true, _y_pred, f_model, label_vocab=corpus_encoder.label_vocab)
            elif args.label_type_dec in {"full-pl-split", "full-pl-split-stat-dyn"}:
                save_preds_encdec_pl(test_corp, _y_true, _y_pred, f_model)
    if not args.save_model:
        classifier.remove(f_model=classifier.f_model)


if __name__ == '__main__':
    main()
