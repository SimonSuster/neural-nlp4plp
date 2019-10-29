import json
import random
from collections import defaultdict

from nlp4plp.evaluate.allpredicates import get_all_predicates, get_full_pl, get_all_predicates_arguments, \
    get_full_pl_no_arg_id, get_full_pl_id, get_full_pl_id_plc
from nlp4plp.evaluate.eval import parse_file

random.seed(1)
import re
from os.path import realpath, join

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

from util import FileUtils, get_file_list, Nlp4plpData, load_json

# beginning of seq, end of seq, beg of line, end of line, unknown, padding symbol
BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<bol>', '</bol>', '<unk>', '<pad>'
STAT = {"property", "group", "size", "given"}
DYN = {"take", "take_wr", "union", "observe", "probability"}

def to_lower(s, low):
    return s.lower() if low else s


class Vocab:
    def __init__(self):
        self.word2idx = dict()  # word to index lookup
        self.idx2word = dict()  # index to word lookup

        self.reserved_sym = dict()  # dictionary of reserved terms with corresponding symbols.

    @classmethod
    def populate_indices(cls, vocab_set, **reserved_sym):
        inst = cls()

        for key, sym in reserved_sym.items():  # populate reserved symbols such as bos, eos, unk, pad
            if sym in vocab_set:
                print("Removing the reserved symbol {} from training corpus".format(sym))
                del vocab_set[sym]  # @todo: delete symbol from embedding space also
            inst.word2idx.setdefault(sym, len(inst.word2idx))  # Add item with given default value if it does not exist.
            inst.reserved_sym[key] = sym  # Populate dictionary of reserved symbols. @todo: check data type of key. Var?
            setattr(cls, key,
                    inst.word2idx[sym])  # Add reserved symbols as class attributes with corresponding idx mapping

        for term in vocab_set:
            inst.word2idx.setdefault(term, len(inst.word2idx))

        inst.idx2word = {val: key for key, val in inst.word2idx.items()}

        return inst

    def __getitem__(self, item):
        return self.word2idx[item]

    def __len__(self):
        return len(self.word2idx)

    @property
    def size(self):
        return len(self.word2idx)

    def to_dict(self):
        return {"reserved": self.reserved_sym,
                'word2idx': [{"key": key, "val": val} for key, val in self.word2idx.items()]}

    @classmethod
    def from_dict(cls, d):
        inst = cls()
        inst.word2idx = {d["key"]: d["val"] for d in
                         d['word2idx']}  # the paramter "d" here is the return value of to_dict function earlier.
        for key, val in d['reserved'].items():
            setattr(inst, key, inst.word2idx[val])
        inst.idx2word = {val: key for key, val in inst.word2idx.items()}

        return inst


def dummy_processor(line):
    return line.strip().split()


# def discretize_labels(fit_labels, transform_labels):
def discretizer(fit_labels, n_bins):
    # sklearn requires a 2D array
    fit_labels = fit_labels.reshape(-1, 1)
    # discretization strategy: uniform | quantile
    # uniform: equal width
    # quantile: equal frequencies (% of the total data, same number of observations per bin)
    # quantile binning is sensitive to the data distribution, which will probably make it perform better
    le = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    le.fit(fit_labels)
    return le
    # return le.transform(transform_labels), le


def encode_labels(fit_labels, transform_labels):
    le = LabelEncoder()
    le.fit(fit_labels)
    print("Label classes: ", list(le.classes_), "respectively mapped to ", le.transform(le.classes_))
    return le.transform(transform_labels), le


class Nlp4plpInst:
    def __init__(self, ls, low=True, tokenizer=word_tokenize, convert_consts=None):
        self.f = None
        self.convert_consts = convert_consts
        self.id, self.ans, self.ans_raw, self.txt, self.statements, self.num2n_map = self.read(ls, low, tokenizer, convert_consts)
        self.words_anno = None
        #self.txt = None
        self.pos = None
        self.f_anno = None

    @staticmethod
    def read(ls, low, tokenize, convert_consts=None):
        problem_l = to_lower(ls[0], low)
        if "\t" in problem_l:
            problem_id = problem_l[1:problem_l.find("\t")].strip()
        else:
            problem_id = problem_l[1:problem_l.find(":")].strip()
        problem_ans_raw = problem_l[problem_l.find("##") + len("## Solution:"):].strip()
        problem_ans_raw = problem_ans_raw.replace("^", "**")
        try:
            problem_ans = eval(problem_ans_raw)
            problem_ans = float(problem_ans)
        except (SyntaxError, NameError):  # capture unsuccessful eval()
            print("can't convert ans to float: {}; problem id: {}".format(
                problem_ans_raw,
                problem_id))
            problem_ans = None
        if ls[1].startswith("%"):
            num2n_map = eval(ls[1].split(" ", 1)[1])
        else:
            num2n_map = None

        rest_ls = ls[1:] if num2n_map is None else ls[2:]
        statements = [to_lower(l.strip(), low) for l in rest_ls if l.strip() and not l.strip().startswith("%")]
        problem_txt_lst = tokenize(problem_l[problem_l.find(":") + 1: problem_l.find("##")].strip())
        # now done in add_txt_anno()
        #if convert_consts == "no-our-map":  # convert num symbols back to numbers in the text based on map
        #    inv_num2n_map = {v: k for k,v in num2n_map.items()}
        #    l = []
        #    for i in problem_txt_lst:
        #        if i in inv_num2n_map:
        #            l.append(inv_num2n_map[i])
        #        else:
        #            l.append(i)
        #problem_txt_lst = l

        # return problem_id, problem_txt_lst, problem_ans, problem_ans_raw, statements
        return problem_id, problem_ans, problem_ans_raw, problem_txt_lst, statements, num2n_map

    def add_txt_anno(self, f):
        """
        Creates a list of toks and a list of PoS tags in the passage based on annotated files
        """

        def dep_list_to_dict(l):
            dep_d = defaultdict(list)
            for d in l:
                if len(re.split("[(-,)]", d)) != 4:
                    continue
                rel, gov, dep, _ = re.split("[(-,)]", d)
                if len(gov.split("-")) > 2:
                    gov = "-".join(gov.split("-")[:2])
                if len(dep.split("-")) > 2:
                    dep = "-".join(dep.split("-")[:2])
                dep_d[dep].append((gov, rel))  # can be multiple parents
            return dep_d

        self.f_anno = f
        fh = load_json(f)
        self.words_anno = fh["words"]
        # synt. dependencies from corenlp
        try:  # not every file processed with corenlp
            dep_l = [i for i in fh["corenlp"] if not (i.startswith("pos(") or i.startswith("word("))]
        except KeyError:
            dep_l = []
        dep_d = dep_list_to_dict(dep_l)
        if self.convert_consts in {"no-our-map", "no"}:  # rewrite the source txt which includes num symbs with original txt with annos
            txt = []
        pos = []
        rels = []
        #rels_second = []  # second order
        num = []
        sen_ns = []  # number of the sentence in which the word occurs
        for i in range(len(self.words_anno)):
            try:
                sen_n = str(i + 1)
                s = self.words_anno[sen_n]
                for j in range(len(s)):
                    if self.convert_consts in {"no-our-map", "no"}:
                        txt.append(s[str(j + 1)]["text"].lower())
                    pos_tag = s[str(j + 1)]["nlp_pos"]
                    pos.append(f"pos:{pos_tag}")
                    dep_id = f"{i + 1}-{j + 1}"
                    gov, rel = "", ""
                    if dep_id in dep_d:
                        for c, (g, r) in enumerate(dep_d[f"{i + 1}-{j + 1}"]):  # glue together feats if > 1
                            gov += f"|{g}" if c > 0 else g
                            rel += f"|{r}" if c > 0 else r
                    rels.append(f"rel:{rel}")
                    try:
                        num_tag = [s[str(j + 1)]["corenlp"]["number"]]
                    except KeyError:
                        num_tag = []
                    num_tag = True if num_tag else False
                    num.append(f"num:{num_tag}")
                    sen_ns.append(f"sen_n:{sen_n}")
            except KeyError:
                continue
        if self.convert_consts in {"no-our-map", "no"}:  # rewrite the source txt which includes num symbs with original txt with annos
            self.txt = txt

        diff_len = len(self.txt) - len(pos)
        if diff_len > 0:
            pos.extend(diff_len*["pos:"])
        self.pos = pos
        diff_len = len(self.txt) - len(rels)
        if diff_len > 0:
            rels.extend(diff_len * ["rel:"])
        self.rels = rels
        diff_len = len(self.txt) - len(num)
        if diff_len > 0:
            num.extend(diff_len * ["num:"])
        self.num = num
        diff_len = len(self.txt) - len(sen_ns)
        if diff_len > 0:
            num.extend(diff_len * ["sen_n:"])
        self.sen_ns = sen_ns


class Nlp4plpCorpus:
    def __init__(self, data_dir, convert_consts):
        self.data_dir = data_dir

        # if 'conv', will convert numbers in statements based on convert.py, but no map will exist for the conversion
        # if 'our-map', will leave the numbers intact in statements. These will be converted based on the map from the
        # problem file, so that when the network outputs symols for numbers, we can convert back to numbers
        # for execution accuracy
        # if 'no-our-map', same as 'our-map', but the numbers in the *text* won't be converted to symbols (but the map will still exist)
        self.convert_consts = convert_consts
        self.fs, self.insts = self.get_insts()
        self.fitted_discretizer = None

    def get_insts(self):
        dir_fs = get_file_list(self.data_dir, [".pl"])
        fs = []  # effective file names after removing corrupt
        insts = []
        for f in dir_fs:
            inst = Nlp4plpInst(Nlp4plpData.read_pl(f), convert_consts=self.convert_consts)
            inst.f = f

            # get anno filename
            inst_au, inst_id = inst.id[0], inst.id[1:]
            if " " in inst_id:
                inst_id = inst_id.split()[0]
            author = {"m": "monica", "l": "liselot", "h": "hannah"}[inst_au]
            dir = "/".join(self.data_dir.split("/")[:-2]) + f"/data/examples/{author}/"
            #dir = f"/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/data/examples/{author}/"
            fn = f"{inst_id:0>10}.json"
            f_anno = dir + fn
            try:
                inst.add_txt_anno(f_anno)
            except FileNotFoundError:
                print(f"anno file {f_anno} not found")

            if inst.ans is not None:
                insts.append(inst)
                fs.append(f)

        print(f"effective num of insts: {len(insts)}")
        return fs, insts

    def shuffle(self):
        assert len(self.fs) == len(self.insts)
        idx = list(range(len(self.fs)))
        np.random.shuffle(idx)
        self.fs = list(np.array(self.fs)[idx])
        self.insts = list(np.array(self.insts)[idx])

    @staticmethod
    def get_from_ids(inst, s_id, t_id):
        # If token index t_id is from sentence 1, we don't need sentence segments.
        # Note that s_id and t_id index from 1, whereas we should use 0-indexing.
        if s_id == 1:
            label = t_id - 1
        else:
            tok_cn = 0
            for i in range(s_id - 1):
                try:
                    tok_cn += len(inst.words_anno[str(i + 1)])
                except KeyError:
                    continue
            label = tok_cn + t_id - 1
        return label

    def get_n_idx_from_ids(self, inst, s_id, t_id, number_txt):
        # Get index of number mentioned in the group part of that token
        # If token index t_id is from sentence 1, we don't need sentence segments.
        # Note that s_id and t_id index from 1, whereas we should use 0-indexing.
        w = inst.words_anno[str(s_id)][str(t_id)]
        try:
            size = w["group"]["size"]
            if str(size["number"]) != number_txt:
                # mistake in annotation
                # find id heuristically
                # print("FIND ID HEURISTICALLY")
                return None
            try:
                s_t_id = size["words"][0]  # what about other list items, where the number needs to be inferred?
            except IndexError:
                # find id heuristically
                # print("FIND ID HEURISTICALLY")
                return None
            hit = re.findall(r"^(\d+)-(\d+)$", s_t_id)
            assert len(hit[0]) == 2
            n_sent_id = int(hit[0][0])
            n_tok_id = int(hit[0][1])
            label_n = self.get_from_ids(inst, n_sent_id, n_tok_id)
        except KeyError:
            return None

        return label_n

    @staticmethod
    def get_from_token(inst, t):
        try:
            label = inst.txt.index(t)
        except ValueError:
            label = None

        return label

    def get_group_label(self, inst):
        """
        group(y)
        """
        gs = [s for s in inst.statements if "group(" in s]  # sent-tok id or just token
        # only use the first one
        g = gs[0]
        attr = re.findall(r"group\((.*)\)", g)[0]
        hit = re.findall(r"^(\d+)-(\d+)$", attr)
        if hit:
            assert len(hit[0]) == 2
            sent_id = int(hit[0][0])
            tok_id = int(hit[0][1])
            label = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label = self.get_from_token(inst, attr)

        return label

    def get_take_label(self, inst):
        """
        take(y1,.,.)
        """
        gs = [s for s in inst.statements if "take(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take\((.*),.*,.*\)", g)[0]
        hit = re.findall(r"^(\d+)-(\d+)$", attr)
        if hit:
            assert len(hit[0]) == 2
            sent_id = int(hit[0][0])
            tok_id = int(hit[0][1])
            label = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label = self.get_from_token(inst, attr)

        return label

    def get_take3_label(self, inst):
        """
        take(.,.,n)
        """
        gs = [s for s in inst.statements if "take(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take\((.*), *(.*), *(.*)\)", g)[0]

        # y2
        hit_y2 = re.findall(r"^(\d+)-(\d+)$", attr[1])
        if hit_y2:
            assert len(hit_y2[0]) == 2
            sent_id = int(hit_y2[0][0])
            tok_id = int(hit_y2[0][1])
            label_y2 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y2 = self.get_from_token(inst, attr)
        if label_y2 is None:
            return None

        # n
        hit_n = re.findall(r"^(\d+)$", attr[2])
        if hit_n:
            # look up the group info from y2
            label_n = self.get_n_idx_from_ids(inst, sent_id, tok_id, hit_n[0])
        else:
            label_n = None
        if label_n is None:
            return None

        return label_n

    def get_take_declen2_label(self, inst):
        """
        take(y1,y2,.)
        """
        gs = [s for s in inst.statements if "take(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take\((.*), *(.*), *.*\)", g)[0]
        labels = []
        for i in attr:
            hit = re.findall(r"^(\d+)-(\d+)$", i)
            if hit:
                assert len(hit[0]) == 2
                sent_id = int(hit[0][0])
                tok_id = int(hit[0][1])
                label = self.get_from_ids(inst, sent_id, tok_id)
            else:
                label = self.get_from_token(inst, attr)
            labels.append(label)
            if label is None:
                return None
        return labels

    def get_take_declen3_label(self, inst):
        """
        take(y1,y2,y3), where y3 is a number whose index in the passage we need to find
        """
        gs = [s for s in inst.statements if "take(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take\((.*), *(.*), *(.*)\)", g)[0]
        labels = []

        # y1
        hit_y1 = re.findall(r"^(\d+)-(\d+)$", attr[0])
        if hit_y1:
            assert len(hit_y1[0]) == 2
            sent_id = int(hit_y1[0][0])
            tok_id = int(hit_y1[0][1])
            label_y1 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y1 = self.get_from_token(inst, attr)
        if label_y1 is None:
            return None
        labels.append(label_y1)

        # y2
        hit_y2 = re.findall(r"^(\d+)-(\d+)$", attr[1])
        if hit_y2:
            assert len(hit_y2[0]) == 2
            sent_id = int(hit_y2[0][0])
            tok_id = int(hit_y2[0][1])
            label_y2 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y2 = self.get_from_token(inst, attr)
        if label_y2 is None:
            return None
        labels.append(label_y2)

        # n
        hit_n = re.findall(r"^(\d+)$", attr[2])
        if hit_n:
            # look up the group info from y2
            label_n = self.get_n_idx_from_ids(inst, sent_id, tok_id, hit_n[0])
        else:
            label_n = None
        if label_n is None:
            return None

        labels.append(label_n)
        return labels

    def get_both_take_declen3_label(self, inst):
        """
        take(y1,y2,y3) & take_wr(y1,y2,y3), where y3 is a number whose index in the passage we need to find
        """
        gs = [s for s in inst.statements if ("take(" in s or "take_wr(" in s)]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"(take|take_wr)\((.*), *(.*), *(.*)\)", g)[0][1:]
        labels = []

        # y1
        hit_y1 = re.findall(r"^(\d+)-(\d+)$", attr[0])
        if hit_y1:
            assert len(hit_y1[0]) == 2
            sent_id = int(hit_y1[0][0])
            tok_id = int(hit_y1[0][1])
            label_y1 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y1 = self.get_from_token(inst, attr)
        if label_y1 is None:
            return None
        labels.append(label_y1)

        # y2
        hit_y2 = re.findall(r"^(\d+)-(\d+)$", attr[1])
        if hit_y2:
            assert len(hit_y2[0]) == 2
            sent_id = int(hit_y2[0][0])
            tok_id = int(hit_y2[0][1])
            label_y2 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y2 = self.get_from_token(inst, attr)
        if label_y2 is None:
            return None
        labels.append(label_y2)

        # n
        hit_n = re.findall(r"^(\d+)$", attr[2])
        if hit_n:
            # look up the group info from y2
            label_n = self.get_n_idx_from_ids(inst, sent_id, tok_id, hit_n[0])
        else:
            label_n = None
        if label_n is None:
            return None

        labels.append(label_n)
        return labels

    def get_take_wr_declen3_label(self, inst):
        """
        take_wr(y1,y2,y3), where y3 is a number whose index in the passage we need to find
        """
        gs = [s for s in inst.statements if "take_wr(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take_wr\((.*), *(.*), *(.*)\)", g)[0]
        labels = []

        # y1
        hit_y1 = re.findall(r"^(\d+)-(\d+)$", attr[0])
        if hit_y1:
            assert len(hit_y1[0]) == 2
            sent_id = int(hit_y1[0][0])
            tok_id = int(hit_y1[0][1])
            label_y1 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y1 = self.get_from_token(inst, attr)
        if label_y1 is None:
            return None
        labels.append(label_y1)

        # y2
        hit_y2 = re.findall(r"^(\d+)-(\d+)$", attr[1])
        if hit_y2:
            assert len(hit_y2[0]) == 2
            sent_id = int(hit_y2[0][0])
            tok_id = int(hit_y2[0][1])
            label_y2 = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label_y2 = self.get_from_token(inst, attr)
        if label_y2 is None:
            return None
        labels.append(label_y2)

        # n
        hit_n = re.findall(r"^(\d+)$", attr[2])
        if hit_n:
            # look up the group info from y2
            label_n = self.get_n_idx_from_ids(inst, sent_id, tok_id, hit_n[0])
        else:
            label_n = None
        if label_n is None:
            return None

        labels.append(label_n)
        return labels

    def get_take_wr_declen2_label(self, inst):
        """
        take_wr(y1,y2,.)
        """
        gs = [s for s in inst.statements if "take_wr(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take_wr\((.*), *(.*), *.*\)", g)[0]
        labels = []
        for i in attr:
            hit = re.findall(r"^(\d+)-(\d+)$", i)
            if hit:
                assert len(hit[0]) == 2
                sent_id = int(hit[0][0])
                tok_id = int(hit[0][1])
                label = self.get_from_ids(inst, sent_id, tok_id)
            else:
                label = self.get_from_token(inst, attr)
            labels.append(label)
            if label is None:
                return None
        return labels

    def get_take_wr_label(self, inst):
        """
        take_wr(y1,.,.)
        """
        gs = [s for s in inst.statements if "take_wr(" in s]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"take_wr\((.*),.*,.*\)", g)[0]
        hit = re.findall(r"^(\d+)-(\d+)$", attr)
        if hit:
            assert len(hit[0]) == 2
            sent_id = int(hit[0][0])
            tok_id = int(hit[0][1])
            label = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label = self.get_from_token(inst, attr)

        return label

    def get_both_take_label(self, inst):
        """
        take(y1,.,.) & take_wr(y1,.,.)
        """
        gs = [s for s in inst.statements if ("take(" in s or "take_wr(" in s)]  # sent-tok id or just token
        if not gs:
            return None
        # only use the first one
        g = gs[0]
        attr = re.findall(r"(take|take_wr)\((.*),.*,.*\)", g)[0][1]
        hit = re.findall(r"^(\d+)-(\d+)$", attr)
        if hit:
            assert len(hit[0]) == 2
            sent_id = int(hit[0][0])
            tok_id = int(hit[0][1])
            label = self.get_from_ids(inst, sent_id, tok_id)
        else:
            label = self.get_from_token(inst, attr)

        return label

    def get_dummy_label(self, inst):
        return 1

    def get_predicates_label(self, inst):
        """
        Get all (outermost) predicate names as labels
        """

        def get_outer_predicates(s):
            hits = re.findall(r"^(\w+)", s)
            assert len(hits) == 1, (s, inst.id)
            return hits.pop()

        labels = []
        for statement in inst.statements:
            predicate = get_outer_predicates(statement)
            labels.append(predicate)

        return labels  # [predicate1, predicate2, ...]

    def get_n_predicates_label(self, inst):
        """
        Get predicate sequence type as a label
        """

        def get_outer_predicates(s):
            hits = re.findall(r"^(\w+)", s)
            assert len(hits) == 1, (s, inst.id)
            return hits.pop()

        predicates = []
        for statement in inst.statements:
            predicates.append(get_outer_predicates(statement))
        label = " ".join(predicates)

        return label

    def get_predicates_all_label(self, inst):
        """
        Get all (outer and inner) predicate names as labels
        """
        problog_program = parse_file(inst.f)
        preds = get_all_predicates(problog_program)
        labels = [p for ps in preds for p in ps]

        return labels  # [predicate1, predicate2, ...]

    def get_predicates_arguments_all_label(self, inst):
        """
        Get all (outer and inner) predicate names as labels
        """
        problog_program = parse_file(inst.f)
        preds = get_all_predicates_arguments(problog_program)
        labels = [p for ps in preds for p in ps]

        return labels  # [predicate1, predicate2, ...]

    def get_full_pl_label(self, inst):
        """
        Includes predicates, arguments, parentheses and dot
        """
        problog_program = parse_file(inst.f)
        preds, id_map = get_full_pl(problog_program, convert_consts=self.convert_consts)
        if self.convert_consts == "no-ent":
            # convert ent symbs back to original
            inv_id_map = {v: k for k, v in id_map.items()}
        labels = []
        for ps in preds:
            for p in ps:
                if self.convert_consts == "conv":
                    labels.append(p)
                elif self.convert_consts in {"our-map", "no-our-map"}:
                    try:
                        mapped_p = inst.num2n_map[p]
                        labels.append(mapped_p)
                    except (KeyError, TypeError):
                        labels.append(p)
                elif self.convert_consts == "no":
                    labels.append(p)
                elif self.convert_consts == "no-ent":
                    if p in inv_id_map:
                        _p = inv_id_map[p]
                        hs = re.findall("(\d)-(\d+)", _p)
                        if hs:
                            s_id, tok_id = hs.pop()
                            try:
                                _p = inst.words_anno[s_id][tok_id]["text"]
                            except KeyError:
                                # wrongly recognized as indices
                                labels.append(_p)
                        labels.append(_p)
                    else:
                        labels.append(p)
                else:
                    raise ValueError
        #numbers that don't get mapped:
        #
        #for l in labels:
        #    try:
        #        f=float(l)
        #        print(f)
        #    except ValueError:
        #        continue
        return labels  # [predicate1, predicate2, ...]

    def get_full_pl_stat_dyn_label(self, inst):
        """
        Includes predicates, arguments, parentheses and dot
        :param str selection: stat | dyn. If None all statements will be used as labels.
                                          If stat, only static, if dyn, only dynamic.
        """
        problog_program = parse_file(inst.f)
        preds, id_map = get_full_pl(problog_program)
        preds1, preds2 = [], []
        for ps in preds:
            start = ps[0]
            p_name = re.findall("(.*?)\(", start).pop()
            if p_name in STAT:
                preds1.append(ps)
            elif p_name in DYN:
                preds2.append(ps)
            else:
                raise ValueError

        labels1 = [p for ps in preds1 for p in ps]
        labels2 = [p for ps in preds2 for p in ps]

        return labels1, labels2  # [predicate1, predicate2, ...]

    def get_full_pl_no_arg_id_label(self, inst):
        """
        Includes predicates, arguments, parentheses and dot, but arg ents don't have ids
        """
        problog_program = parse_file(inst.f)

        if self.convert_consts == "conv":
            preds = get_full_pl_no_arg_id(problog_program, self.convert_consts)
            labels = [p for ps in preds for p in ps]
        elif self.convert_consts in {"our-map", "no-our-map"}:
            preds, id_map = get_full_pl(problog_program, convert_consts=self.convert_consts)
            labels = []
            for ps in preds:
                for p in ps:
                    if p in inst.num2n_map:
                        labels.append("n")
                    elif re.match("^l\d+$", p):
                        labels.append("l")
                    else:
                        labels.append(p)
        else:
            raise ValueError

        return labels  # [predicate1, predicate2, ...]
        #problog_program = parse_file(inst.f)
        #preds = get_full_pl_no_arg_id(problog_program, self.convert_consts)
        #labels = [p for ps in preds for p in ps]

        #return labels  # [predicate1, predicate2, ...]

    def get_full_pl_id_label(self, inst):
        """
        Instead of args ents we have ids (integers), instead of all other elements we have COPY
        """
        problog_program = parse_file(inst.f)
        preds = get_full_pl_id(problog_program, self.convert_consts)
        labels = [p for ps in preds for p in ps]

        return labels

    def get_full_pl_id_plc_label(self, inst):
        """
        Instead of args ents we have ids (integers), instead of all other elements we have COPY
        """
        problog_program = parse_file(inst.f)
        preds = get_full_pl_id_plc(problog_program)
        labels = [p for ps in preds for p in ps]

        return labels

    def get_compact_pl_id_plc_label(self, inst):
        """
        Instead of args ents we have ids (integers), instead of all other elements we have COPY
        """
        problog_program = parse_file(inst.f)
        preds = get_full_pl_id_plc(problog_program, compact=True, convert_consts=self.convert_consts)
        labels = [p for ps in preds for p in ps]

        return labels

    def get_pointer_labels(self, label_type):
        if label_type == "group":
            get_label = self.get_group_label
        elif label_type == "take":
            get_label = self.get_take_label
        elif label_type == "take_wr":
            get_label = self.get_take_wr_label
        elif label_type == "both_take":
            get_label = self.get_both_take_label
        elif label_type == "take3":
            get_label = self.get_take3_label
        elif label_type == "take_declen2":
            get_label = self.get_take_declen2_label
        elif label_type == "take_wr_declen2":
            get_label = self.get_take_wr_declen2_label
        elif label_type == "take_declen3":
            get_label = self.get_take_declen3_label
        elif label_type == "take_wr_declen3":
            get_label = self.get_take_wr_declen3_label
        elif label_type == "both_take_declen3":
            get_label = self.get_both_take_declen3_label
        elif label_type == "dummy":
            get_label = self.get_dummy_label
        else:
            raise ValueError("invalid label_type specified")

        for inst in self.insts:
            inst.label = get_label(inst)

    def get_labels(self, label_type, max_output_len=None):
        if label_type == "predicates":
            get_label = self.get_predicates_label
        elif label_type == "n-predicates":
            get_label = self.get_n_predicates_label
        elif label_type == "predicates-all":
            get_label = self.get_predicates_all_label
        elif label_type == "predicates-arguments-all":
            get_label = self.get_predicates_arguments_all_label
        elif label_type == "full-pl":
            get_label = self.get_full_pl_label
        elif label_type == "full-pl-no-arg-id":
            get_label = self.get_full_pl_no_arg_id_label
        #elif label_type == "full-pl-id":
        #    get_label = self.get_full_pl_id_label
        elif label_type == "full-pl-split":
            get_label = (self.get_full_pl_no_arg_id_label, self.get_full_pl_id_label)
        elif label_type == "full-pl-split-plc":
            #get_label = (self.get_full_pl_no_arg_id_label, self.get_full_pl_id_plc_label)
            get_label = (self.get_full_pl_no_arg_id_label, self.get_compact_pl_id_plc_label)
        elif label_type == "full-pl-split-stat-dyn":
            get_label = self.get_full_pl_stat_dyn_label
        else:
            raise ValueError("invalid label_type specified")

        for inst in self.insts:
            if label_type == "full-pl-split-stat-dyn":
                inst.label, inst.label2 = get_label(inst)
                if max_output_len is not None:
                    if len(inst.label) > max_output_len:
                        inst.label = None
                    if len(inst.label2) > max_output_len:
                        inst.label2 = None
            elif isinstance(get_label, tuple):
                inst.label = get_label[0](inst)
                inst.label2 = get_label[1](inst)
                assert len(inst.label) == len(inst.label2)
                if max_output_len is not None:
                    if len(inst.label) > max_output_len:
                        inst.label = None
                    if len(inst.label2) > max_output_len:
                        inst.label2 = None
            elif label_type == "n-predicates":
                inst.ans_discrete = get_label(inst)
            else:
                inst.label = get_label(inst)
                if max_output_len is not None and len(inst.label) > max_output_len:
                    inst.label = None

    def remove_none_labels(self):
        n_before = len(self.insts)
        new_insts = []
        new_fs = []
        for f, inst in zip(self.fs, self.insts):
            if inst.label is not None:
                new_insts.append(inst)
                new_fs.append(f)
        self.insts = new_insts
        self.fs = new_fs
        n_after = len(self.insts)
        print(f"{n_before - n_after} instances removed (label is None)")

    def add_tok_ids(self, c=1):
        for inst in self.insts:
            inst.tok_ids = list(range(c, c + len(inst.txt)))
            c += len(inst.txt)

        return c

    def discretize(self, n_bins=None, fitted_discretizer=None):
        """
        use a discretizer fitted on train data for dev/test; for train, fit from scratch
        """
        anss = np.array([inst.ans for inst in self.insts]).reshape(-1, 1)
        if fitted_discretizer is None:
            fitted_discretizer = discretizer(anss, n_bins)
        anss_discrete = fitted_discretizer.transform(anss)
        anss_discrete = anss_discrete.flatten()
        for ans_discrete, inst in zip(anss_discrete, self.insts):
            inst.ans_discrete = ans_discrete
        self.fitted_discretizer = fitted_discretizer


class Corpus:
    def __init__(self, dir_corpus, f_labels, dir_labels, fname_subset, text_processor=dummy_processor,
                 label_encoder=encode_labels):
        self.dir_in = dir_corpus
        self.fname_subset = fname_subset  # file names for the current split of the corpus

        all_labels = FileUtils.read_json(f_labels, dir_labels)
        self.labels = [all_labels[i] for i in self.fname_subset]
        all_labels = list(all_labels.values())
        self.labels, self.label_encoder = label_encoder(all_labels, self.labels)

        self.text_processor = text_processor

    def __iter__(self):
        for cur_fname, cur_label in zip(self.fname_subset, self.labels):
            with open(realpath(join(self.dir_in, cur_fname + '.txt'))) as f:
                word_seq = list()
                for line in f:
                    word_seq.extend(self.text_processor(line))
                yield (word_seq, cur_label)


class CorpusEncoder:

    def __init__(self, vocab, label_vocab=None, label_vocab2=None):  # , skipgrams):
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.label_vocab2 = label_vocab2

    @classmethod
    def from_corpus(cls, *corpora):
        # create vocab set for initializing Vocab class
        vocab_set = set()

        for corpus in corpora:
            for (words, labels) in corpus:
                for word in words:
                    if not word in vocab_set:
                        vocab_set.add(word)

        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),

        return cls(vocab)

    def encode_inst(self, inst):
        '''
        Converts sentence to sequence of indices after adding beginning, end and replacing unk tokens.
        @todo: check if beg and end of seq and line are required for our classification setup.
        '''
        out = [self.transform_item(i) for i in inst]
        # if self.vocab.bos is not None:
        #     out = [self.vocab.bos] + out
        # if self.vocab.eos is not None:
        #     out = out + [self.vocab.eos]
        return out

    def encode_label(self, label):
        '''
        Converts sentence to sequence of indices after adding beginning, end and replacing unk tokens.
        @todo: check if beg and end of seq and line are required for our classification setup.
        '''
        out = self.transform_label(label)
        # if self.vocab.bos is not None:
        #     out = [self.vocab.bos] + out
        return out

    def transform_label(self, label):
        # return self.label_vocab.word2idx[label]
        try:
            return self.label_vocab.word2idx[label]
        except KeyError:
            if self.label_vocab.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.label_vocab.unk

    def transform_item(self, item):
        '''
        Returns the index for an item if present in vocab, <unk> otherwise.
        '''
        try:
            return self.vocab.word2idx[item]
        except KeyError:
            if self.vocab.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.vocab.unk

    def get_batches(self, corpus, batch_size):

        instances = list()
        labels = list()

        for inst in corpus.insts:
            cur_inst = self.encode_inst(inst.txt)
            instances.append(cur_inst)
            if isinstance(inst.ans_discrete, str):
                labels.append(self.encode_label(inst.ans_discrete))
            else:
                labels.append(inst.ans_discrete)
            if len(instances) == batch_size:
                yield (instances, labels)
                instances = list()
                labels = list()

        if instances:
            yield (instances, labels)

    def batch_to_tensors(self, cur_insts, cur_labels, device):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + self.vocab.pad  # this creates a tensor of padding indices

        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))

        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)
        labels = torch.LongTensor(cur_labels).to(device)

        return t, labels, lengths

    def decode_inst(self, inst):
        out = [self.vocab.idx2word[i] for i in inst if i != self.vocab.pad]
        return out

    def replace_unk(self, inst):
        out = [self.vocab.idx2word[self.vocab.unk] if i not in self.vocab.word2idx else i for i in inst]
        return out

    def get_decoded_sequences(self, corpus, strip_angular=False):
        instances = list()

        for (cur_inst, __) in iter(corpus):
            cur_inst = self.replace_unk(cur_inst)
            if strip_angular:
                # stripping angular brackets to support HTML rendering
                cur_inst = [i.strip('<>') for i in cur_inst]
            instances.append(cur_inst)

        return instances

    def to_json(self, fname, dir_out):
        with open(realpath(join(dir_out, fname)), 'w') as f:
            json.dump({'vocab': self.vocab.to_dict()}, f)

    @classmethod
    def from_json(cls, fname, dir_out):
        with open(realpath(join(dir_out, fname))) as f:
            obj = json.load(f)

        vocab = Vocab.from_dict(obj['vocab'])

        return cls(vocab)


class Nlp4plpEncoder(CorpusEncoder):
    @classmethod
    def from_corpus(cls, *corpora):
        # create vocab set for initializing Vocab class
        vocab_set = set()
        # sg_set = set()
        label_vocab_set = set()

        for corpus in corpora:
            for inst in corpus.insts:
                # sg_set.add(skipgrams(words, n = 3, k = 1))
                for word in inst.txt:
                    if not word in vocab_set:
                        vocab_set.add(word)
                if inst.ans_discrete not in label_vocab_set:
                    label_vocab_set.add(inst.ans_discrete)
        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),
        # sg = Vocab.populate_indices(sg_set)
        label_vocab = Vocab.populate_indices(label_vocab_set, eos=EOS, pad=PAD, unk=UNK)

        # return cls(vocab, sg)
        return cls(vocab, label_vocab)

    @classmethod
    def feature_from_corpus(cls, *corpora, feat_type=["pos", "rels", "num", "sen_ns"]):
        # create vocab set for initializing Vocab class
        vocab_set = set()
        # sg_set = set()

        for corpus in corpora:
            for inst in corpus.insts:
                for f_t in feat_type:
                    feat = getattr(inst, f_t)
                    feat_set = set(feat)
                    vocab_set.update(feat_set)
        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),
        # sg = Vocab.populate_indices(sg_set)

        # return cls(vocab, sg)
        return cls(vocab)

    def get_feature_batches(self, corpus, batch_size, feat_type):
        instances = list()
        for inst in corpus.insts:
            cur_inst = [self.encode_inst(getattr(inst, f_t)) for f_t in feat_type]
            # feats = [self.encode_inst(getattr(inst, f_t)) for f_t in feat_type]
            # cur_inst = list(zip(*feats))
            instances.append(cur_inst)
            if len(instances) == batch_size:
                yield instances
                instances = list()

        if instances:
            yield instances

    def feature_batch_to_tensors(self, cur_insts, device, n_feat_types):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        # lengths = [len(inst) for inst in cur_insts]
        lengths = [len(inst[0]) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        t = torch.zeros(n_inst, n_feat_types, maxlen,
                        dtype=torch.int64) + self.vocab.pad  # this creates a tensor of padding indices

        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            for c, i in enumerate(inst):  # several feat types
                t[idx, c, :length].copy_(torch.tensor(i))

        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)

        return t, lengths  # t: b*n_feat_types*maxlen


class Nlp4plpRegressionEncoder(Nlp4plpEncoder):
    def get_batches(self, corpus, batch_size):

        instances = list()
        labels = list()

        for inst in corpus.insts:
            cur_inst = self.encode_inst(inst.txt)
            instances.append(cur_inst)
            labels.append(inst.ans)
            if len(instances) == batch_size:
                yield (instances, labels)
                instances = list()
                labels = list()

        if instances:
            yield (instances, labels)

    def batch_to_tensors(self, cur_insts, cur_labels, device):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + self.vocab.pad  # this creates a tensor of padding indices

        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))

        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)
        labels = torch.FloatTensor(cur_labels).to(device)

        return t, labels, lengths


class Nlp4plpPointerNetEncoder(Nlp4plpEncoder):
    def get_batches(self, corpus, batch_size):
        instances = list()
        labels = list()
        for inst in corpus.insts:
            cur_inst = self.encode_inst(inst.txt)
            instances.append(cur_inst)
            labels.append(inst.label)
            if len(instances) == batch_size:
                yield (instances, labels)
                instances = list()
                labels = list()

        if instances:
            yield (instances, labels)

    def batch_to_tensors(self, cur_insts, cur_labels, device):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + self.vocab.pad  # this creates a tensor of padding indices

        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))

        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)
        labels = torch.LongTensor(cur_labels).to(device)

        return t, labels, lengths


class Nlp4plpEncDecEncoder(CorpusEncoder):
    @classmethod
    def from_corpus(cls, *corpora):
        # create vocab set for initializing Vocab class
        vocab_set = set()
        label_vocab_set = set()

        for corpus in corpora:
            for inst in corpus.insts:
                for word in inst.txt:
                    if word not in vocab_set:
                        vocab_set.add(word)
                for label in inst.label:
                    if label not in label_vocab_set:
                        label_vocab_set.add(label)

        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),
        label_vocab = Vocab.populate_indices(label_vocab_set, eos=EOS, pad=PAD, unk=UNK)

        return cls(vocab, label_vocab)

    def encode_labels(self, labels):
        '''
        Converts sentence to sequence of indices after adding beginning, end and replacing unk tokens.
        @todo: check if beg and end of seq and line are required for our classification setup.
        '''
        out = [self.transform_label(label) for label in labels]
        # if self.vocab.bos is not None:
        #     out = [self.vocab.bos] + out
        if self.vocab.eos is not None:
            out = out + [self.vocab.eos]
        return out

    def transform_label(self, label):
        # return self.label_vocab.word2idx[label]
        try:
            return self.label_vocab.word2idx[label]
        except KeyError:
            if self.label_vocab.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.label_vocab.unk

    def get_batches(self, corpus, batch_size, token_ids=False):
        instances = list()
        labels = list()
        for inst in corpus.insts:
            if token_ids:
                cur_inst = inst.tok_ids
            else:
                cur_inst = self.encode_inst(inst.txt)
            instances.append(cur_inst)
            cur_labels = self.encode_labels(inst.label)
            labels.append(cur_labels)
            if len(instances) == batch_size:
                yield (instances, labels)
                instances = list()
                token_ids = list()
                labels = list()

        if instances:
            yield (instances, labels)

    def get_feature_batches(self, corpus, batch_size, feat_type=["pos"]):
        instances = list()
        for inst in corpus.insts:
            cur_inst = self.encode_inst(inst.pos)
            instances.append(cur_inst)
            if len(instances) == batch_size:
                yield instances
                instances = list()

        if instances:
            yield instances

    def batch_to_tensors(self, cur_insts, cur_labels, device, padding_idx):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)
        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + padding_idx  # this creates a tensor of padding indices
        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))
        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)

        label_lengths = [len(label) for label in cur_labels]
        n_inst, maxlen = len(cur_labels), max(label_lengths)
        label_t = torch.zeros(n_inst, maxlen,
                              dtype=torch.int64) + self.label_vocab.pad  # this creates a tensor of padding indices
        # copy the sequence
        for idx, (label, length) in enumerate(zip(cur_labels, label_lengths)):
            label_t[idx, :length].copy_(torch.tensor(label))
        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        label_t = label_t.contiguous().to(device)
        label_lengths = torch.tensor(label_lengths, dtype=torch.int).to(device)

        return t, lengths, label_t, label_lengths

    def feature_batch_to_tensors(self, cur_insts, device):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + self.vocab.pad  # this creates a tensor of padding indices

        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))

        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)

        return t, lengths

    def strip_until_eos(self, labels):
        """
        Keep the labels in decoded sequence up to EOS symbol
        :param labels: list of label arrays
        """
        new_labels = []
        for arr in labels:
            idxs = np.where(arr == self.label_vocab.eos)[0]
            if idxs.size != 0:
                new_labels.append(arr[:idxs[0]])
            else:
                new_labels.append(arr)

        return new_labels


class Nlp4plpEncSplitDecEncoder(Nlp4plpEncDecEncoder):
    @classmethod
    def from_corpus(cls, *corpora):
        # create vocab set for initializing Vocab class
        vocab_set = set()
        label_vocab_set = set()
        label_vocab2_set = set()

        for corpus in corpora:
            for inst in corpus.insts:
                for word in inst.txt:
                    if word not in vocab_set:
                        vocab_set.add(word)
                for label in inst.label:
                    if label not in label_vocab_set:
                        label_vocab_set.add(label)
                for label in inst.label2:
                    if label not in label_vocab2_set:
                        label_vocab2_set.add(label)

        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),
        label_vocab = Vocab.populate_indices(label_vocab_set, eos=EOS, pad=PAD, unk=UNK)
        label_vocab2 = Vocab.populate_indices(label_vocab2_set, eos=EOS, pad=PAD, unk=UNK)

        return cls(vocab, label_vocab, label_vocab2)

    def get_batches(self, corpus, batch_size, token_ids=False):
        instances = list()
        labels = list()
        labels2 = list()
        for inst in corpus.insts:
            if token_ids:
                cur_inst = inst.tok_ids
            else:
                cur_inst = self.encode_inst(inst.txt)
            instances.append(cur_inst)
            cur_labels = self.encode_labels(inst.label)
            cur_labels2 = self.encode_labels2(inst.label2)
            labels.append(cur_labels)
            labels2.append(cur_labels2)
            if len(instances) == batch_size:
                yield (instances, labels, labels2)
                instances = list()
                token_ids = list()
                labels = list()
                labels2 = list()

        if instances:
            yield (instances, labels, labels2)

    def encode_labels2(self, labels):
        '''
        Converts sentence to sequence of indices after adding beginning, end and replacing unk tokens.
        @todo: check if beg and end of seq and line are required for our classification setup.
        '''
        out = [self.transform_label2(label) for label in labels]
        # if self.vocab.bos is not None:
        #     out = [self.vocab.bos] + out
        if self.vocab.eos is not None:
            out = out + [self.vocab.eos]
        return out

    def transform_label2(self, label):
        # return self.label_vocab.word2idx[label]
        try:
            return self.label_vocab2.word2idx[label]
        except KeyError:
            if self.label_vocab2.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.label_vocab2.unk

    def batch_to_tensors(self, cur_insts, cur_labels, cur_labels2, device, padding_idx):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)
        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + padding_idx  # this creates a tensor of padding indices
        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))
        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype=torch.int).to(device)

        label_lengths = [len(label) for label in cur_labels]
        n_inst, maxlen = len(cur_labels), max(label_lengths)
        label_t = torch.zeros(n_inst, maxlen,
                              dtype=torch.int64) + self.label_vocab.pad  # this creates a tensor of padding indices
        # copy the sequence
        for idx, (label, length) in enumerate(zip(cur_labels, label_lengths)):
            label_t[idx, :length].copy_(torch.tensor(label))
        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        label_t = label_t.contiguous().to(device)
        label_lengths = torch.tensor(label_lengths, dtype=torch.int).to(device)

        label_lengths2 = [len(label) for label in cur_labels2]
        n_inst, maxlen = len(cur_labels2), max(label_lengths2)
        label_t2 = torch.zeros(n_inst, maxlen,
                              dtype=torch.int64) + self.label_vocab2.pad  # this creates a tensor of padding indices
        # copy the sequence
        for idx, (label, length) in enumerate(zip(cur_labels2, label_lengths2)):
            label_t2[idx, :length].copy_(torch.tensor(label))
        # contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        label_t2 = label_t2.contiguous().to(device)
        label_lengths2 = torch.tensor(label_lengths2, dtype=torch.int).to(device)

        return t, lengths, label_t, label_lengths, label_t2, label_lengths2

class DataUtils:

    @staticmethod
    def split_data(f_labels, dir_labels, dir_splits):
        labels = FileUtils.read_json(f_labels, dir_labels)
        sorted_labels = sorted(labels.items())
        f_idx = [i for i, j in sorted_labels]  # file names in sorted order
        label_list = [j for i, j in sorted_labels]  # labels sorted according to file names

        train_split, val_split, test_split = DataUtils.create_splits(f_idx, label_list)

        FileUtils.write_list(train_split, 'train_ids.txt', dir_splits)
        FileUtils.write_list(val_split, 'val_ids.txt', dir_splits)
        FileUtils.write_list(test_split, 'test_ids.txt', dir_splits)

        return (train_split, val_split, test_split)

    @staticmethod
    def create_splits(doc_ids, labels):
        train_idx, rest_idx, __, rest_labels = train_test_split(doc_ids, labels, stratify=labels, test_size=0.2)
        val_idx, test_idx = train_test_split(rest_idx, stratify=rest_labels, test_size=0.5)

        return (train_idx, val_idx, test_idx)

    @staticmethod
    def read_splits(dir_splits):
        train_split = FileUtils.read_list('train_ids.txt', dir_splits)
        val_split = FileUtils.read_list('val_ids.txt', dir_splits)
        test_split = FileUtils.read_list('test_ids.txt', dir_splits)

        return (train_split, val_split, test_split)


def get_bert_embs(insts, bert_client, bert_embs={}):
    #  e = bert_client.encode(['First do it', 'then do it right', 'then do it better'])
    #  e[0] --> first sentence
    #  e[0][0] --> CLS
    #  e[0][1] --> w0
    #  ...
    #  e[0][3] -->  w2
    #  e[0][4] --> SEP
    #  e[0][5] --> zero embedding for padding

    # for inst in self.insts:
    #    print(inst)
    embs = bert_client.encode([" ".join(inst.txt) for inst in insts])
    for inst, emb in zip(insts, embs):  # embs for one instance
        # get token emb
        for i, tok_i in enumerate(inst.tok_ids):
            bert_embs[tok_i] = emb[1:][i]
    return bert_embs


def get_max_nsymb_batches(corpus, batch_size):
    max_nsymbs = list()
    for inst in corpus.insts:
        # find max n symbol for masking
        n_symbs = [int(v[1:]) for v in inst.num2n_map.values()]
        max_nsymb = max(n_symbs) if n_symbs else None
        max_nsymbs.append(max_nsymb)
        if len(max_nsymbs) == batch_size:
            yield max_nsymbs
            max_nsymbs = list()

    if max_nsymbs:
        yield max_nsymbs
