import argparse
import os
import re
import sys
from collections import Counter
from shutil import copyfile

import numpy as np

from corpus_util import Nlp4plpCorpus
from util import get_file_list

ORD2NUM = {"first": "1",
           "second": "2",
           "third": "3",
           "fourth": "4",
           "fifth": "5",
           "sixth": "6",
           "seventh": "7",
           "eighth": "8",
           "nineth": "9",
           "tenth": "10",
           "eleventh": "11",
           "twelfth": "12"}

def create_splits(data_dir, data_dir_out, seed=1234):
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

    np.random.seed(seed)
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


def get_ans_dist(data):
    """
    :param data: a list of Nlp4plpInst objects
    """
    dist = Counter([d.ans for d in data])

    return dist


def get_vocab_dist(data):
    """
    :param data: a list of Nlp4plpInst objects
    """
    dist = Counter([w for d in data for w in d.txt])

    return dist


def get_avg_sen_len(data):
    return np.mean([len(d.txt) for d in data])


def main(data_dir):
    d = Nlp4plpCorpus(data_dir)
    insts = d.insts

    ans_dist = get_ans_dist(insts)
    print(f"Most common answers: {ans_dist.most_common(10)}")

    vocab_dist = get_vocab_dist(insts)
    print(f"Most common vocab items: {vocab_dist.most_common(10)}")
    print(f"Vocab size: {len(vocab_dist)}")
    print(f"Dataset (problem/text) size: {sum(vocab_dist.values())}")
    avg_sen_len = get_avg_sen_len(insts)
    print(f"Avg sen len: {avg_sen_len}")
    # create_splits(args.data_dir, "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/")


def norm_nums_txt(inst, all_set):
    # normalize
    txt = []
    for i in range(len(inst.words_anno)):
        try:
            s = inst.words_anno[str(i + 1)]
            for j in range(len(s)):
                #pos_tag = s[str(j + 1)]["nlp_pos"]
                w = s[str(j + 1)]["text"].lower()
                num = None
                try:
                    num = str(s[str(j + 1)]["corenlp"]["number"])
                    all_set.add(w)
                except KeyError:
                    # "percent"
                    if w == "percent":
                        num = "100"
                        all_set.add(w)
                    # "7th"
                    elif re.findall("\d+th", w):
                        h = re.findall("(\d+)th", w)
                        num = h.pop()
                        all_set.add(w)
                    # ordinals ("third" etc.)
                    elif w in ORD2NUM:
                        num = ORD2NUM[w]
                        all_set.add(w)
                    # "a", "an"
                    #elif w in {"a", "an"}:
                    #    num = "1"
                    #    all_set.add(w)
                txt.append(num if num is not None else w)
        except KeyError:
            continue

    return txt, all_set


def map_nums_txt(txt):
    # map
    mapped_txt = []
    num2n_map = {}
    for w in txt:
        h = re.findall("^\.?\d+\.?\d*$", w)
        if h:
            _h = h.pop()
            if _h in num2n_map:
                mapped_txt.append(num2n_map[_h])
            else:
                num2n_map[_h] = f"n{len(num2n_map)}"
                mapped_txt.append(num2n_map[_h])
        else:
            mapped_txt.append(w)

    # difficult cases where number is not explicitly mentioned but needs to be inferred:
    #if "1" not in num2n_map and re.findall("1[\,\)]", "\n".join(inst.statements)):
    #    print("...")
    return mapped_txt, num2n_map


def write_f(mapped_txt, inst, num2n_map, f_out):
    statements_txt = "\n".join(inst.statements)
    mapped_txt = " ".join(mapped_txt)
    with open(f_out, "w") as fh:
        fh.write(f"% {inst.id}: {mapped_txt} ## Solution= {inst.ans_raw}\n")
        fh.write(f"% {num2n_map}\n\n")
        fh.write(f"{statements_txt}\n")


def convert_nums(data_dir, out_data_dir):
    d = Nlp4plpCorpus(data_dir)
    all_set = set()

    for f, inst in zip(d.fs, d.insts):
        # convert
        txt, all_set = norm_nums_txt(inst, all_set)
        mapped_txt, num2n_map = map_nums_txt(txt)

        # write
        f_out = out_data_dir + os.path.basename(f)
        write_f(mapped_txt, inst, num2n_map, f_out)
    #print("\n".join(sorted(list(all_set))))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--data-dir", type=str,
                            default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/",
                            help="path to folder from where data is loaded")
    arg_parser.add_argument("--out-data-dir", type=str,
                            default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/",
                            help="path to save the mapped data")
    args = arg_parser.parse_args()

    # main(args.data_dir)
    convert_nums(f"{args.data_dir}train/", f"{args.out_data_dir}train/")
    convert_nums(f"{args.data_dir}dev/", f"{args.out_data_dir}dev/")
    convert_nums(f"{args.data_dir}test/", f"{args.out_data_dir}test/")
