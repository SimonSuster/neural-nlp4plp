import json
import re
from os.path import realpath, join

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

from util import FileUtils, get_file_list, Nlp4plpData, load_json

# beginning of seq, end of seq, beg of line, end of line, unknown, padding symbol
BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<bol>', '</bol>', '<unk>', '<pad>'


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
    def __init__(self, ls, low=True, tokenizer=word_tokenize):
        self.id, self.ans, self.ans_raw, self.statements = self.read(ls, low, tokenizer)
        self.words_anno = None
        self.txt = None
        self.f_anno = None

    @staticmethod
    def read(ls, low, tokenize):
        problem_l = to_lower(ls[0], low)
        if "\t" in problem_l:
            problem_id = problem_l[1:problem_l.find("\t")].strip()
        else:
            problem_id = problem_l[1:problem_l.find(":")].strip()
        #problem_txt_lst = tokenize(problem_l[problem_l.find(":") + 1:problem_l.find("##")].strip())
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
        statements = [to_lower(l.strip(), low) for l in ls[1:] if l.strip()]

        #return problem_id, problem_txt_lst, problem_ans, problem_ans_raw, statements
        return problem_id, problem_ans, problem_ans_raw, statements

    def add_txt_anno(self, f):
        fh = load_json(f)
        self.words_anno = fh["words"]
        self.f_anno = f
        txt = []
        for i in range(len(self.words_anno)):
            try:
                s = self.words_anno[str(i + 1)]
                for j in range(len(s)):
                    txt.append(s[str(j + 1)]["text"].lower())
            except KeyError:
                continue
        self.txt = txt


class Nlp4plpCorpus:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fs, self.insts = self.get_insts()
        self.fitted_discretizer = None

    def get_insts(self):
        dir_fs = get_file_list(self.data_dir, [".pl"])
        fs = []  # effective file names after removing corrupt
        insts = []
        for f in dir_fs:
            inst = Nlp4plpInst(Nlp4plpData.read_pl(f))

            # get anno filename
            inst_au, inst_id = inst.id[0], inst.id[1:]
            if " " in inst_id:
                inst_id = inst_id.split()[0]
            author = {"m": "monica", "l": "liselot", "h": "hannah"}[inst_au]
            dir = "/".join(self.data_dir.split("/")[:5]) + f"/data/examples/{author}/"
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
                    tok_cn += len(inst.words_anno[str(i+1)])
                except KeyError:
                    continue
            label = tok_cn + t_id
        return label

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

    def get_pointer_labels(self, label_type):
        if label_type == "group":
            get_label = self.get_group_label
        elif label_type == "take":
            get_label = self.get_take_label
        elif label_type == "take_wr":
            get_label = self.get_take_wr_label
        elif label_type == "both_take":
            get_label = self.get_both_take_label
        elif label_type == "take_declen2":
            get_label = self.get_take_declen2_label
        elif label_type == "take_wr_declen2":
            get_label = self.get_take_wr_declen2_label
        elif label_type == "dummy":
            get_label = self.get_dummy_label
        else:
            raise ValueError("invalid label_type specified")

        for inst in self.insts:
            inst.pointer_label = get_label(inst)

    def remove_none_labels(self):
        n_before = len(self.insts)
        new_insts = []
        new_fs = []
        for f, inst in zip(self.fs, self.insts):
            if inst.pointer_label is not None:
                new_insts.append(inst)
                new_fs.append(f)
        self.insts = new_insts
        self.fs = new_fs
        n_after = len(self.insts)
        print(f"{n_before - n_after} instances removed (pointer_label is None)")

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

    def __init__(self, vocab):  # , skipgrams):
        self.vocab = vocab
        # self.sg = skipgrams

    @classmethod
    def from_corpus(cls, *corpora):
        # create vocab set for initializing Vocab class
        vocab_set = set()
        # sg_set = set()

        for corpus in corpora:
            for (words, labels) in corpus:
                # sg_set.add(skipgrams(words, n = 3, k = 1))
                for word in words:
                    if not word in vocab_set:
                        vocab_set.add(word)

        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),
        # sg = Vocab.populate_indices(sg_set)

        # return cls(vocab, sg)
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

        for corpus in corpora:
            for inst in corpus.insts:
                # sg_set.add(skipgrams(words, n = 3, k = 1))
                for word in inst.txt:
                    if not word in vocab_set:
                        vocab_set.add(word)

        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)  # bos=BOS, eos=EOS, bol=BOL, eol=EOL),
        # sg = Vocab.populate_indices(sg_set)

        # return cls(vocab, sg)
        return cls(vocab)


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
            labels.append(inst.pointer_label)
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
