import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, accuracy_score

from corpus_util import Nlp4plpCorpus, Nlp4plpRegressionEncoder
from util import load_emb


class NNEmb(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim, word_idx, pretrained_emb_path):
        super(NNEmb, self).__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path

        self.device = torch.device('cuda')

        assert pretrained_emb_path is not None
        self.word_embeddings, dim = load_emb(pretrained_emb_path, word_idx, freeze=False)
        assert dim == self.embedding_dim

        self.cos = nn.CosineSimilarity(dim=1)

    def load_train_embs(self, corpus):
        # avoid batching to avoid zero padding
        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, len(corpus.insts))):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)
            word_embs = self.word_embeddings(cur_insts).to(self.device)  # inst length * n insts * dim
            insts_embs = word_embs.mean(0)
        assert idx == 0

        return insts_embs, cur_labels

    def predict(self, corpus, corpus_encoder, train_embs, y_train):
        train_size = len(train_embs)
        y_pred = list()
        y_true = list()

        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, len(corpus.insts))):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)
            y_true.extend(cur_labels.cpu().numpy())

            word_embs = self.word_embeddings(cur_insts).to(self.device)  # inst length * n insts * dim
            insts_embs = word_embs.mean(0)

            for inst_emb in insts_embs:
                inst_emb_list = [inst_emb] * train_size
                inst_emb_temp = torch.stack(inst_emb_list)
                sims = self.cos(inst_emb_temp, train_embs)
                score, inst_idx = torch.max(sims, 0)
                ans = y_train[inst_idx]
                y_pred.append(np.asscalar(ans.cpu().numpy()))
        assert idx == 0

        assert len(y_pred) == len(y_true)
        return y_pred, y_true


class RandomPointer:
    def predict(self, corpus, corpus_encoder, train_embs, y_train):
        y_true = []
        y_pred = []

        for inst in corpus.insts:
            y_true.append(inst.label)
            y_pred.append(np.random.randint(len(inst.txt)))

        return y_pred, y_true


class PosRandomPointer:
    def predict(self, corpus, corpus_encoder, train_embs, y_train):
        """
        Randomly pick a noun from the first sentence in the passage
        """
        y_true = []
        y_pred = []

        for inst in corpus.insts:
            y_true.append(inst.label)
            pool = [w["index"] - 1 for w in inst.words_anno['1'].values() if w["nlp_pos"].startswith("NN")]
            if pool:
                pred = np.random.choice(pool)
            else:
                pred = np.random.randint(len(inst.txt))
            y_pred.append(pred)
        return y_pred, y_true


class SamplingPointer:
    def __init__(self, train_corpus):
        # get group object pointers and resolve them to words
        y_true = []
        for inst in train_corpus.insts:
            if isinstance(inst.label, (list, np.ndarray)):
                w = [inst.txt[i] for i in inst.label]
            else:
                w = inst.txt[inst.label]
            y_true.append(w)

        if isinstance(w, list):
            self.y_probs_keys = []
            self.y_probs_vals = []
            length = len(w)
            for i in range(length):
                y_counts = Counter([l[i] for l in y_true])
                total = sum(y_counts.values())
                y_probs = {k: v / total for k, v in y_counts.items()}
                self.y_probs_keys.append(list(y_probs.keys()))
                self.y_probs_vals.append(list(y_probs.values()))
        else:
            y_counts = Counter(y_true)
            total = sum(y_counts.values())
            y_probs = {k: v / total for k, v in y_counts.items()}
            self.y_probs_keys = list(y_probs.keys())
            self.y_probs_vals = list(y_probs.values())

    def predict(self, corpus, corpus_encoder, train_embs, y_train):
        """
        Randomly pick a noun from the first sentence in the passage
        """
        y_true = []
        y_pred = []

        for inst in corpus.insts:
            y_true.append(inst.label)
            if isinstance(inst.label, (list, np.ndarray)):
                length = len(inst.label)
                pred_l = []  # contains length-many predictions
                for j in range(length):
                    sample = np.random.choice(self.y_probs_keys[j], len(self.y_probs_keys[j]), p=self.y_probs_vals[j], replace=False)
                    i = 0
                    pred = None
                    # check most probable words first
                    while i < len(sample):
                        if sample[i] in inst.txt:
                            pred = inst.txt.index(sample[i])
                            break
                        i += 1
                    # random pointer if sampling failed
                    if pred is None:
                        pred = np.random.randint(len(inst.txt))
                    pred_l.append(pred)
                y_pred.append(pred_l)
            else:
                # call this once only?
                sample = np.random.choice(self.y_probs_keys, len(self.y_probs_keys), p=self.y_probs_vals, replace=False)
                i = 0
                pred = None
                # check most probable words first
                while i < len(sample):
                    if sample[i] in inst.txt:
                        pred = inst.txt.index(sample[i])
                        break
                    i += 1
                # random pointer if sampling failed
                if pred is None:
                    pred = np.random.randint(len(inst.txt))
                y_pred.append(pred)

        return y_pred, y_true


class RandomProb:
    def predict(self, corpus, corpus_encoder, train_embs, y_train):
        y_true = []
        y_pred = []
        for inst in corpus.insts:
            y_true.append(inst.ans)
            y_pred.append(np.random.rand())

        return y_pred, y_true


class AvgProb(object):
    def __init__(self, corpus):
        self.avg_prob = np.mean([inst.ans for inst in corpus.insts])

    def predict(self, corpus, corpus_encoder, train_embs, y_train):
        y_true = []
        y_pred = []
        for inst in corpus.insts:
            y_true.append(inst.ans)
            y_pred.append(self.avg_prob)

        return y_pred, y_true


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--data-dir", type=str,
                            default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/",
                            help="path to folder from where data is loaded")
    arg_parser.add_argument("--embed-size", type=int, default=50, help="embedding dimension")
    arg_parser.add_argument("--model", type=str, help="nearest-neighbour-emb | pos-random-pointer | sampling-pointer | random-prob | avg-prob")
    arg_parser.add_argument("--n-runs", type=int, default=50, help="number of runs to average over the results")
    arg_parser.add_argument("--pretrained-emb-path", type=str,
                            help="path to the txt file with word embeddings")
    arg_parser.add_argument("--pointer-type", type=str,
                            help="group | take | take_declen3", default="group")
    args = arg_parser.parse_args()

    train_corp = Nlp4plpCorpus(args.data_dir + "train")
    test_corp = Nlp4plpCorpus(args.data_dir + "test")

    test_score_runs = []

    for n in range(args.n_runs):
        if args.model == "nearest-neighbour-emb":
            eval_score = mean_absolute_error
            corpus_encoder = Nlp4plpRegressionEncoder.from_corpus(train_corp)
            classifier_params = {'vocab_size': corpus_encoder.vocab.size,
                                 'padding_idx': corpus_encoder.vocab.pad,
                                 'embedding_dim': args.embed_size,
                                 'word_idx': corpus_encoder.vocab.word2idx,
                                 'pretrained_emb_path': args.pretrained_emb_path
                                 }

            classifier = NNEmb(**classifier_params)
            train_embs, y_train = classifier.load_train_embs(train_corp)
        elif args.model == "random-pointer":
            eval_score = accuracy_score
            label_type = args.pointer_type
            test_corp = Nlp4plpCorpus(args.data_dir + "test")
            test_corp.get_pointer_labels(label_type=label_type)
            test_corp.remove_none_labels()
            corpus_encoder = None
            train_embs = None
            y_train = None
            classifier = RandomPointer()
        elif args.model == "pos-random-pointer":
            eval_score = accuracy_score
            label_type = args.pointer_type
            test_corp = Nlp4plpCorpus(args.data_dir + "test")
            test_corp.get_pointer_labels(label_type=label_type)
            test_corp.remove_none_labels()
            corpus_encoder = None
            train_embs = None
            y_train = None
            classifier = PosRandomPointer()
        elif args.model == "sampling-pointer":
            eval_score = accuracy_score
            label_type = args.pointer_type
            train_corp = Nlp4plpCorpus(args.data_dir + "train")
            train_corp.get_pointer_labels(label_type=label_type)
            train_corp.remove_none_labels()
            test_corp = Nlp4plpCorpus(args.data_dir + "test")
            test_corp.get_pointer_labels(label_type=label_type)
            test_corp.remove_none_labels()
            corpus_encoder = None
            train_embs = None
            y_train = None
            classifier = SamplingPointer(train_corp)
        elif args.model == "random-prob":
            eval_score = mean_absolute_error
            test_corp = Nlp4plpCorpus(args.data_dir + "test")
            corpus_encoder = None
            train_embs = None
            y_train = None
            classifier = RandomProb()
        elif args.model == "avg-prob":
            eval_score = mean_absolute_error
            train_corp = Nlp4plpCorpus(args.data_dir + "train")
            test_corp = Nlp4plpCorpus(args.data_dir + "test")
            corpus_encoder = None
            train_embs = None
            y_train = None
            classifier = AvgProb(train_corp)
            print(f"avg prob: {classifier.avg_prob}")
        else:
            raise NotImplementedError

        # get predictions
        y_pred, y_true = classifier.predict(test_corp, corpus_encoder, train_embs, y_train)
        if isinstance(y_pred[0], list):
            y_true = [str(y) for y in y_true]
            y_pred = [str(y) for y in y_pred]

        test_acc = eval_score(y_true=y_true, y_pred=y_pred)
        test_score_runs.append(test_acc)
        print('TEST SCORE: %.3f' % test_acc)
    print('AVG TEST SCORE over %d runs: %.3f' % (args.n_runs, np.mean(test_score_runs)))
