import argparse
import random
from datetime import datetime

from comet_ml import Experiment

random.seed(1)

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)

from sklearn.metrics import accuracy_score

from corpus_util import Nlp4plpCorpus, Nlp4plpEncoder, Nlp4plpEncDecEncoder, RerankerData
from net import EncoderDecoder, LSTMClassifier


def idx_to_labels(_y_pred, _y_true, idx2word):
    labels_pred = []
    for i in _y_pred:
        labels_beam = []
        for cand in i:
            labels_beam.append([idx2word[idx] for idx in cand])
        labels_pred.append(labels_beam)
    labels_true = []
    for i in _y_true:
        labels_true.append([idx2word[idx] for idx in i])

    return labels_pred, labels_true


def main():
    arg_parser = argparse.ArgumentParser(description="reranker for the base model for neural-nlp4plp")
    arg_parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    arg_parser.add_argument("--beam-decoding", action="store_true")
    arg_parser.add_argument("--beam-topk", type=int, default=10)
    arg_parser.add_argument("--beam-width", type=int, default=10)
    arg_parser.add_argument("--bidir", action="store_true")
    arg_parser.add_argument("--convert-consts", type=str, default="no", help="conv | our-map | no-our-map | no. \n/"
                                                                             "conv-> txt: -; stats: num_sym+ent_sym.\n/"
                                                                             "our-map-> txt: num_sym; stats: num_sym(from map)+ent_sym;\n/"
                                                                             "no-our-map-> txt: -; stats: num_sym(from map)+ent_sym;\n/"
                                                                             "no-> txt: -; stats: -, only ent_sym;\n/"
                                                                             "no-ent-> txt: -; stats: -, no ent_sym;\n/")
    arg_parser.add_argument("--cuda", type=int, default=0, help="train on GPU, default: 0")
    arg_parser.add_argument("--data-dir", type=str, default="",
                            help="path to folder from where data is loaded. Subfolder should be train/dev/test")
    arg_parser.add_argument("--debug", action="store_true")
    arg_parser.add_argument("--embed-size", type=int, default=50, help="embedding dimension")
    arg_parser.add_argument('--feat-type', nargs='+', help="Which feature to use: pos | rels | num | sen_ns")
    arg_parser.add_argument("--feat_embed-size", type=int, default=10, help="embedding dimension for external features")
    arg_parser.add_argument("--hidden-dim", type=int, default=50, help="")
    arg_parser.add_argument("--label-type-dec", type=str, default="full-pl",
                            help="predicates | n-predicates | n-full | predicates-all | predicates-arguments-all | full-pl | full-pl-no-arg-id | full-pl-split | full-pl-split-plc | full-pl-split-stat-dyn. To use with EncDec.")
    arg_parser.add_argument("--log-experiment", action="store_true", help="logs using Comet.ML")
    arg_parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default: 0.001")
    model_names = "lstm-enc-discrete-dec | lstm-enc-dec | lstm-enc-split-dec"
    arg_parser.add_argument("--max-output-len", type=int, default=200,
                            help="Maximum decoding length for EncDec models at prediction time.")
    arg_parser.add_argument("--model", type=str, default="lstm-enc-dec", help=f"")
    arg_parser.add_argument("--model-path", type=str, help=f"saved model file to load")
    arg_parser.add_argument("--n-layers", type=int, default=1, help="number of layers for the RNN")
    arg_parser.add_argument('--rank-feat-type', nargs='+', help="Which feature to use: TBD")
    arg_parser.add_argument('--rank-discrete-feat-type', nargs='+', help="Which feature to use: score | rank. Score uses beam candidate score as a feature, rank uses the beam rank position.")
    arg_parser.add_argument("--ret-period", type=int, default=20,
                            help="stop training if no improvement in both acc and f1 during last 'ret-period' epochs")
    arg_parser.add_argument("--save-model", action="store_true")

    args = arg_parser.parse_args()
    assert "nums_mapped" in args.data_dir

    print(args)
    # initialize corpora to get base-model predictions on training
    if args.model in {"lstm-enc-dec", "lstm-enc-split-dec", "lstm-enc-discrete-dec"}:
        train_corp = Nlp4plpCorpus(args.data_dir + "train", args.convert_consts)
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

        train_corp.remove_none_labels()
        dev_corp.remove_none_labels()
        test_corp.remove_none_labels()
    else:
        raise ValueError(f"Model should be '{model_names}'")

    # keep track of comet experiment instances
    experiments = []

    # load model
    if args.model == 'lstm-enc-dec':
        model_path = args.model_path
        # experiment not yet defined
        experiment = Experiment(project_name="neural-nlp4plp", disabled=not args.log_experiment)

        corpus_encoder = Nlp4plpEncDecEncoder.from_corpus(train_corp, dev_corp)

        if args.feat_type:
            feature_encoder = Nlp4plpEncoder.feature_from_corpus(train_corp, dev_corp, feat_type=args.feat_type)
        eval_score = accuracy_score
        classifier = EncoderDecoder.load(f_model=model_path)
        experiment.log_parameters({"n_layers": classifier.n_lstm_layers,
                                   "hidden_dim": classifier.hidden_dim,
                                   "vocab_size": classifier.vocab_size,
                                   "padding_idx": classifier.padding_idx,
                                   "label_padding_idx": classifier.label_padding_idx,
                                   "embedding_dim": classifier.emb_dim,
                                   "dropout": classifier.dropout,
                                   "batch_size": classifier.batch_size,
                                   "word_idx": classifier.word_idx,
                                   "label_idx": classifier.label_idx,
                                   "pretrained_emb_path": classifier.pretrained_emb_path,
                                   "max_output_len": classifier.max_output_len,
                                   "label_size": classifier.n_labels,
                                   "f_model": classifier.f_model,
                                   "bidir": classifier.bidir,
                                   "bert_embs": classifier.bert_embs,
                                   "constrained_decoding": classifier.constrained_decoding,
                                   "feature_idx": classifier.feature_idx,
                                   "feat_size": classifier.feat_size,
                                   "feat_padding_idx": classifier.feat_padding_idx,
                                   "feat_emb_dim": classifier.feat_emb_dim,
                                   "feat_type": classifier.feat_type,
                                   "feat_onehot": classifier.feat_onehot,
                                   "cuda": classifier.cuda})
        experiment.set_model_graph(repr(classifier))

        assert classifier.word_idx == corpus_encoder.vocab.word2idx
        assert classifier.label_idx == corpus_encoder.label_vocab.word2idx, (
            classifier.label_idx, corpus_encoder.label_vocab.word2idx)
    else:
        raise ValueError(f"Model should be '{model_names}'")

    # get base-model predictions on train, dev, test
    _y_pred_train, _y_pred_train_scores, _y_true_train = classifier.predict(train_corp, feature_encoder, corpus_encoder, False,
                                                      args.beam_decoding, args.beam_width,
                                                      args.beam_topk)  # len(train) * beam size
    _y_pred_dev, _y_pred_dev_scores, _y_true_dev = classifier.predict(dev_corp, feature_encoder, corpus_encoder, False,
                                                  args.beam_decoding, args.beam_width,
                                                  args.beam_topk)  # len(train) * beam size
    _y_pred_test, _y_pred_test_scores, _y_true_test = classifier.predict(test_corp, feature_encoder, corpus_encoder, False,
                                                    args.beam_decoding, args.beam_width,
                                                    args.beam_topk)  # len(train) * beam size

    # map idx to labels
    labels_pred_train, labels_true_train = idx_to_labels(_y_pred_train, _y_true_train,
                                                         corpus_encoder.label_vocab.idx2word)
    labels_pred_dev, labels_true_dev = idx_to_labels(_y_pred_dev, _y_true_dev,
                                                     corpus_encoder.label_vocab.idx2word)
    labels_pred_test, labels_true_test = idx_to_labels(_y_pred_test, _y_true_test,
                                                       corpus_encoder.label_vocab.idx2word)

    # prepare reranking train, dev and test data
    rank_train_corp = RerankerData(labels_pred_train, _y_pred_train_scores, labels_true_train, train_corp.fs)
    print(f"Size of reranking train: {len(rank_train_corp.insts)}")
    rank_dev_corp = RerankerData(labels_pred_dev, _y_pred_dev_scores, labels_true_dev, dev_corp.fs)
    print(f"Size of reranking dev: {len(rank_dev_corp.insts)}")
    rank_test_corp = RerankerData(labels_pred_test, _y_pred_test_scores, labels_true_test, test_corp.fs)
    print(f"Size of reranking test: {len(rank_test_corp.insts)}")

    rank_train_corp.remove_empty_txt()
    rank_dev_corp.remove_empty_txt()
    rank_test_corp.remove_empty_txt()

    rank_train_corp.stat()
    rank_dev_corp.stat()
    rank_test_corp.stat()

    # save memory
    del train_corp
    del dev_corp
    del test_corp
    del classifier
    del corpus_encoder
    del feature_encoder

    # prepare vocab encoder
    rank_corpus_encoder = Nlp4plpEncoder.from_corpus(rank_train_corp, rank_dev_corp)
    print(f"n ranking labels: {len(rank_corpus_encoder.label_vocab)}")
    print(rank_corpus_encoder.label_vocab.word2idx)
    rank_feature_encoder = None

    # logging stuff
    f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    print(f"f_model: {f_model}")

    # ranking model init
    net_params = {'n_layers': args.n_layers,
                  'hidden_dim': args.hidden_dim,
                  'vocab_size': rank_corpus_encoder.vocab.size,
                  'padding_idx': rank_corpus_encoder.vocab.pad,
                  'label_padding_idx': -100,
                  'embedding_dim': args.embed_size,
                  'dropout': 0.,
                  'label_size': len(rank_corpus_encoder.label_vocab),
                  'batch_size': args.batch_size,
                  'word_idx': rank_corpus_encoder.vocab.word2idx,
                  'label_idx': rank_corpus_encoder.label_vocab.word2idx,
                  'pretrained_emb_path': None,
                  'f_model': f_model,
                  'bidir': args.bidir,
                  'cuda': args.cuda
                  }
    if args.rank_feat_type:
        feature_encoder = Nlp4plpEncoder.feature_from_corpus(train_corp, dev_corp, feat_type=args.rank_feat_type)
        net_params['feature_idx'] = feature_encoder.vocab.word2idx
        net_params['feat_size'] = feature_encoder.vocab.size
        net_params['feat_padding_idx'] = feature_encoder.vocab.pad
        net_params['feat_emb_dim'] = args.feat_embed_size
        net_params['feat_type'] = args.feat_type
        net_params['feat_onehot'] = args.feat_onehot
    if args.rank_discrete_feat_type:
        net_params['discrete_feat_type'] = args.rank_discrete_feat_type


    classifier = LSTMClassifier(**net_params)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    best_dev_score, experiment = classifier.train_model(rank_train_corp, rank_dev_corp, rank_corpus_encoder,
                                                        rank_feature_encoder, None, args.ret_period, optimizer,
                                                        experiment, return_scores=True)
    experiments.append(experiment)


if __name__ == '__main__':
    main()
