import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--data-dir", type=str,
                            default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/",
                            help="path to folder from where data is loaded")
    arg_parser.add_argument("--embed-size", type=int, default=50, help="embedding dimension")
    arg_parser.add_argument("--model", type=str, help="nearest-neighbour-emb")
    arg_parser.add_argument("--pretrained-emb-path", type=str,
                            help="path to the txt file with word embeddings")
    args = arg_parser.parse_args()

    train_corp = Nlp4plpCorpus(args.data_dir + "train")
    test_corp = Nlp4plpCorpus(args.data_dir + "test")

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
    # get predictions
    y_pred, y_true = classifier.predict(test_corp, corpus_encoder, train_embs, y_train)
    test_acc = eval_score(y_true=y_true, y_pred=y_pred)
    print('TEST SCORE: %.3f' % test_acc)
