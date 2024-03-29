from __future__ import print_function

import json
import os
import random
from collections import Counter

random.seed(0)
from os import makedirs
from os.path import exists, join, realpath

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)
from torch import nn

long_tensor_type = torch.LongTensor
float_tensor_type = torch.FloatTensor

if (torch.cuda.is_available()):
    long_tensor_type = torch.cuda.LongTensor
    float_tensor_type = torch.cuda.FloatTensor


class TorchUtils:

    @staticmethod
    def get_sort_unsort(lengths):
        _, sort = torch.sort(lengths, descending=True)
        _, unsort = sort.sort()
        return sort, unsort

    @staticmethod
    def save_model(state, fname_state, dir_state):
        '''
        Save model state along with relevant architecture parameters as a state dictionary
        :param state: state dictionary with relevant details (e.g. network arch, epoch, model states and optimizer states)
        :param fname_state: out file name
        :param dir_state: out directory
        '''
        if not exists(dir_state):
            makedirs(dir_state)

        # serialize model state
        torch.save(state, realpath(join(dir_state, fname_state)))

    @staticmethod
    def load_model(fname_state, dir_state):
        '''
        Load dictionary of model state and arch params
        :param fname_state: state file name to load
        :param dir_state: directory with filename
        '''
        if not exists(realpath(join(dir_state, fname_state))):
            raise FileNotFoundError("Model not found")

        # load model state
        state = torch.load(realpath(join(dir_state, fname_state)))

        return state

    @staticmethod
    def _set_eval(model):
        '''
        Switch off the training behaviour of the parameters
        '''
        for p in model.parameters():
            p.train = False


class FileUtils:
    @staticmethod
    def write_json(obj_dict, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            json.dump(obj_dict, f)

    @staticmethod
    def read_json(fname, dir_in):
        with open(realpath(join(dir_in, fname))) as f:
            return json.load(f)

    @staticmethod
    def write_list(data_list, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            for term in data_list:
                f.write(term + '\n')

    @staticmethod
    def read_list(fname, dir_in):

        data = list()
        with open(realpath(join(dir_in, fname))) as f:
            for line in f:
                data.append(line.strip())

        return data


class Nlp4plpData:
    @staticmethod
    def read_pl(fpath):
        with open(realpath(fpath)) as f:
            return f.readlines()


def get_file_list(topdir, identifiers=None, all_levels=False):
    """
    :param identifiers: a list of strings, any of which should be in the filename
    :param all_levels: get filenames recursively
    """
    if identifiers is None:
        identifiers = [""]
    filelist = []
    for root, dirs, files in os.walk(topdir):
        if not all_levels and (root != topdir):  # don't go deeper
            continue
        for filename in files:
            get = False
            for i in identifiers:
                if i in filename:
                    get = True
            if get:
                fullname = os.path.join(root, filename)
                filelist.append(fullname)

    return filelist


def load_bert(bert_embs, emb_dim=1024, freeze=False):
    vocab_size = max(bert_embs.keys())+1
    e_m = np.random.normal(size=(vocab_size, emb_dim), loc=0, scale=0.1)
    for i, e in bert_embs.items():
        e_m[i] = e
    embs_tensor = nn.Embedding.from_pretrained(float_tensor_type(e_m), freeze=freeze)

    return embs_tensor, emb_dim


def load_emb(fname, word_idx, freeze=False):
    pretr_embs, pretr_emb_idx, n = load_w2v(fname)
    # build rep. for entities by averaging word vectors
    embs = np.random.normal(size=(len(word_idx) + 1, n), loc=0, scale=0.1)
    embs = update_vectors(pretr_embs, pretr_emb_idx, embs, word_idx)
    embs_tensor = nn.Embedding.from_pretrained(float_tensor_type(embs), freeze=freeze)

    return embs_tensor, n


def load_w2v(fn):
    emb_idx = {}
    with open(fn) as fh:
        m, n = map(eval, fh.readline().strip().split())
        e_m = np.random.normal(size=(m, n), loc=0, scale=0.1)
        for c, l in enumerate(fh):
            w, *e = l.strip().split()
            if len(e) != n:
                print("Incorrect embedding dimension, skipping.")
                continue
            if not w or not e:
                print("Empty w or e.")
            emb_idx[w] = c
            e_m[c] = e
    return e_m, emb_idx, n


def update_vectors(pretr_embs, pretr_emb_idx, embs, word_idx):
    c = 0
    for w, i in word_idx.items():
        if w not in pretr_emb_idx:
            continue
        embs[i] = pretr_embs[pretr_emb_idx[w]]
    return embs


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def f1_score(y_true, y_pred, output_pr=False):
    common = Counter(y_pred) & Counter(y_true)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(y_pred)
    recall = 1.0 * num_same / len(y_true)
    f1 = (2 * precision * recall) / (precision + recall)

    if output_pr:
        return f1, precision, recall
    else:
        return f1


def mean_reciprocal_rank(rs):
    """https://gist.github.com/bwhite/3726239

    Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def save_bert_repr():
    pass
