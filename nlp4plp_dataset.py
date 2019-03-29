import argparse
import os
import sys
from collections import Counter
from shutil import copyfile

import numpy as np

from corpus_util import Nlp4plpCorpus


def get_ans_dist(data):
    """
    :param data: a list of Nlp4plpInst objects
    """
    dist = Counter([d.ans for d in data])

    return dist


def create_splits(data_dir, data_dir_out):
    train_out_dir = data_dir_out + "/train/"
    dev_out_dir = data_dir_out + "/dev/"
    test_out_dir = data_dir_out + "/test/"

    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    elif os.listdir(train_out_dir):
        sys.exit("dir not empty")
    if not os.path.exists(dev_out_dir):
        os.makedirs(dev_out_dir)
    elif os.listdir(dev_out_dir):
        sys.exit("dir not empty")
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)
    elif os.listdir(test_out_dir):
        sys.exit("dir not empty")

    corp = Nlp4plpCorpus(data_dir)
    corp_len = len(corp.fs)

    np.random.seed(1234)
    np.random.shuffle(corp.fs)
    test_len = round(.1 * corp_len)
    val_len = test_len
    test = corp.fs[:test_len]
    dev = corp.fs[test_len:(test_len + val_len)]
    train = corp.fs[(test_len + val_len):]

    for f in test:
        copyfile(f, test_out_dir + os.path.basename(f))
    for f in dev:
        copyfile(f, dev_out_dir + os.path.basename(f))
    for f in train:
        copyfile(f, train_out_dir + os.path.basename(f))

    print(f"Split {corp_len} files into {data_dir_out}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--data-dir", type=str,
                            default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples/",
                            help="path to folder from where data is loaded")
    args = arg_parser.parse_args()

    d=Nlp4plpCorpus(args.data_dir)
    insts = d.insts
    ans_dist = get_ans_dist(insts).most_common()
    print(ans_dist)

    #create_splits(args.data_dir, "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/")
