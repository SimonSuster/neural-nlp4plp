from collections import Counter

import numpy as np

from corpus_util import Nlp4plpCorpus
from util import load_json, f1_score


class JsonPred:
    def __init__(self, f_json):
        self.f_json = f_json
        self.preds = load_json(f_json)
        # {'fn': {'true': ['l1', ...], 'pred': ['l1', ...], 'correct': bool}}

    def get_pos_neg(self):
        correct = {}
        incorrect = {}
        for fn, pred in self.preds.items():
            if pred["correct"]:
                correct[fn] = pred
            else:
                incorrect[fn] = pred

        return correct, incorrect

    def acc_per_len(self):
        len_to_n_cor = Counter()
        len_to_n_incor = Counter()
        for pred in self.preds.values():
            if pred["correct"]:
                len_to_n_cor[len(pred["pred"])] += 1
            else:
                len_to_n_incor[len(pred["pred"])] += 1
        print(sorted(len_to_n_cor.items()))
        print(sorted(len_to_n_incor.items()))

    def inspect_sorted_f1(self):
        """Get predictions sorted on f1 (only incorrect ones)"""
        scores = [(f1_score(y_true=pred["true"], y_pred=pred["pred"]), fn, pred) for fn, pred in self.preds.items()]
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        for i in range(len(scores)):
            if scores[i][0] == 1.:
                continue
            else:
                s = i
                break

        return scores[s:]


def print_sorted_f1(insts, max_n=30):
    for c,i in enumerate(insts):
        if c == max_n:
            break
        both = []
        for t,p in zip(i[2]["true"], i[2]["pred"]):
            if t==p:
                both.append(f"({t}, {p})")
            else:
                both.append(f"(**** {t} ****, **** {p} ****)")
        print(both)
        print(i[1])


def acc_vs_len(correct, incorrect):
    mean_correct = np.mean([len(i["true"]) for i in correct.values()])
    mean_incorrect = np.mean([len(i["true"]) for i in incorrect.values()])
    print(f"mean label TRUE seq len correct: {mean_correct}")
    print(f"mean label TRUE seq len incorrect: {mean_incorrect}")

    mean_correct = np.mean([len(i["pred"]) for i in correct.values()])
    mean_incorrect = np.mean([len(i["pred"]) for i in incorrect.values()])
    print(f"mean label PRED seq len correct: {mean_correct}")
    print(f"mean label PRED seq len incorrect: {mean_incorrect}")


def preds_in_gold(correct, train_corp):
    """
    Out of those got correct, how many are found in its exact (or near-exact?) form in the training set
    :param correct:
    :return:
    """
    train_labels = {" ".join(i.label) for i in train_corp.insts}
    preds = [" ".join(c["pred"]) for c in correct.values()]
    in_train = 0

    for pred in preds:
        if pred in train_labels:
            in_train += 1

    print(f"{len(set(preds))} diff labels predicted (out of {len(preds)} predictions)")
    print(f"{in_train} predictions out of {len(preds)} were in train")


if __name__ == '__main__':
    #jp = JsonPred(f_json="../out/20190621_101845_634116.json")  # outermost
    #label_type_dec = "predicates"
    jp = JsonPred(f_json="../out/20190621_102143_175948.json")  # bidir, all preds+args
    label_type_dec = "predicates-arguments-all"
    correct, incorrect = jp.get_pos_neg()
    insts = jp.inspect_sorted_f1()
    print_sorted_f1(insts)
    acc_vs_len(correct, incorrect)
    #jp.acc_per_len()

    #train_corp = Nlp4plpCorpus("/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/train")

    #max_output_len = 100 if label_type_dec == "predicates-arguments-all" else 50
    #train_corp.get_labels(label_type=label_type_dec, max_output_len=max_output_len)
    #train_corp.remove_none_labels()
    #preds_in_gold(correct, train_corp)
