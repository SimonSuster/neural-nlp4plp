import re
from collections import Counter

import numpy as np

from corpus_util import Nlp4plpCorpus
from util import load_json, f1_score
from sklearn.metrics import accuracy_score, mean_absolute_error


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


def solver_report(f):
    """
    :param f: output of 'bash run_all.sh',
        e.g.:
        === Test h678  ===
        Solution: sat(atleast(3,l3,l2)): 0.85296443
        Time: 1.107825369

    :return: dict
    """
    solver_dict = {"solved": {}, "errors": {}}
    with open(f) as fh:
        l0 = fh.readline()
        while l0:
            assert l0.startswith("===")
            id = l0.split(" ")[2]
            if id == "m435":
                print()
            l1 = fh.readline()
            if not l1.startswith("Solution"):
                print()
            if "Timeout exceeded" in l1:
                solver_dict["errors"][id] = "Timeout exceeded"
            elif re.findall("Solution:.*: \d(.\d*)?", l1):
                try:
                    prob = float(re.findall("Solution:.*: (\d(\..*)?)", l1)[0][0])
                    assert 0. <= prob <= 1.
                    solver_dict["solved"][id] = prob
                except ValueError:
                    print(el1)
            else:
                try:
                    _, el0, el1 = l1.split(":", 2)
                    el0 = el0.strip()
                    solver_dict["errors"][id] = el0
                except ValueError:
                    solver_dict["errors"][id] = "No output"

            l2 = fh.readline()
            assert l2.startswith("Time")

            l0 = fh.readline()

    stat = Counter(solver_dict["errors"].values())
    return solver_dict, stat


def main_solver():
    #solver_output_f = "/home/suster/Apps/out/log_w_gold_test/solver_output_pl_t"
    #test_corp = Nlp4plpCorpus(
    #    "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "test",
    #    convert_consts="no")

    # solver_output_f = "/home/suster/Apps/out/log_w20191008_170849_062092/solver_output_pl_p"
    # test_corp = Nlp4plpCorpus("/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "dev",
    #    convert_consts="our-map")

    # solver_output_f = "/home/suster/Apps/out/log_w20191015_145853_447894/solver_output_pl_p"
    # test_corp = Nlp4plpCorpus("/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "dev", convert_consts="no-our-map")

    #solver_output_f = "/home/suster/Apps/out/log_w20191015_161430_083999/solver_output_pl_p"
    #test_corp = Nlp4plpCorpus(
    #    "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "dev",
    #    convert_consts="no")

    # solver_output_f = "/home/suster/Apps/out/log_w20191017_124408_738282/solver_output_pl_p"
    # test_corp = Nlp4plpCorpus(
    #    "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "dev",
    #    convert_consts="no-our-map")

    # solver_output_f = "/home/suster/Apps/out/log_w20191018_143339_894302/solver_output_pl_p"
    # test_corp = Nlp4plpCorpus(
    #    "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "dev", convert_consts="no")

    #solver_output_f = "/home/suster/Apps/out/log_wnearest-neighbour-full-pl_testset/solver_output_pl_p"
    #test_corp = Nlp4plpCorpus(
    #    "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "test", convert_consts="no")

    solver_output_f = "/home/suster/Apps/out/log_w20191028_154505_649365/solver_output_pl_p"
    test_corp = Nlp4plpCorpus(
        "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits_nums_mapped/" + "test", convert_consts="our-map")



    test_dict = {inst.id: inst.ans for inst in test_corp.insts}
    solver_dict, stat = solver_report(solver_output_f)
    print(f"N errors: {len(solver_dict['errors'])}")
    print(f"N solved: {len(solver_dict['solved'])}")
    print(stat)
    common_ids = test_dict.keys() & solver_dict["solved"].keys()
    true = []
    pred = []
    for id in common_ids:
        true.append(test_dict[id])
        pred.append(solver_dict["solved"][id])
    print(f"mae: {mean_absolute_error(true, pred):.3f}")
    true_str = [str(round(i, 4)) for i in true]
    pred_str = [str(round(i, 4)) for i in pred]
    acc = accuracy_score(true_str, pred_str)
    print(f"acc: {acc:.3f}")
    n_correct = acc*len(solver_dict['solved'])
    corr_acc = n_correct / (len(solver_dict['errors']) + len(solver_dict['solved']))
    print(f"corr acc: {corr_acc:.3f}")

    # non-zero output
    common_ids = test_dict.keys() & {k for k, v in solver_dict["solved"].items() if v != 0}
    print(f'{len(common_ids)} out of {len(solver_dict["solved"].items())} solved had non-zero output.')
    true = []
    pred = []
    for id in common_ids:
        true.append(test_dict[id])
        pred.append(solver_dict["solved"][id])
    print(f"mae (non-zero): {mean_absolute_error(true, pred):.3f}")
    true_str = [str(round(i, 4)) for i in true]
    pred_str = [str(round(i, 4)) for i in pred]
    print(f"acc (non-zero): {accuracy_score(true_str, pred_str):.3f}")


def main_test_train_comparison():
    #jp = JsonPred(f_json="../out/20190621_101845_634116.json")  # outermost
    #label_type_dec = "predicates"
    jp = JsonPred(f_json="../out/20190621_102143_175948.json")  # bidir, all preds+args
    label_type_dec = "predicates-arguments-all"
    correct, incorrect = jp.get_pos_neg()
    insts = jp.inspect_sorted_f1()
    print_sorted_f1(insts)
    acc_vs_len(correct, incorrect)
    jp.acc_per_len()

    train_corp = Nlp4plpCorpus("/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/examples_splits/train")

    max_output_len = 100 if label_type_dec == "predicates-arguments-all" else 50
    train_corp.get_labels(label_type=label_type_dec, max_output_len=max_output_len)
    train_corp.remove_none_labels()
    preds_in_gold(correct, train_corp)

if __name__ == '__main__':
    main_solver()

    #main_test_train_comparison()


