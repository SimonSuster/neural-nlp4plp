import argparse
import os
import re
from collections import Counter

from util import get_file_list

pred_arity = {
    "group": 1,
    "size": 2,
    "given": 1,
    "take": 3,
    "take_wr": 3,
    "union": 2,
    "observe": 1,
    "probability": 1
}


def iter_p_a(d):
    """
    iterator over predicate and args in a line (statement)
    """
    for s in d:
        p, a = s.split("(", 1)
        a = a.rsplit(")", 1)[0]
        yield p, a


class ConstraintStats:
    def __init__(self, data):
        self.data = data
        self.c = Counter()
        self.c_f = Counter()
        n = len(data)

    def __str__(self):
        return "\n".join(f'{k}: {v}' for k, v in self.c.items())

    def get_all(self):
        self.c["N_is_integer"] = Counter()
        for d in self.data:
            c = self.arity(d)
            self.c["arity"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.N_is_integer(d)
            self.c["N_is_integer"] += c
            if len(c) != 0:
                self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += sum(c.values())

            c = self.all_attr_vals_diff(d)
            self.c["all_attr_vals_diff"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.repeated_statements(d)
            self.c["repeated_statements"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.repeated_property_vals(d)
            self.c["repeated_property_vals"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            #self.c["givens_for_group"] += self.givens_for_group(d)

            c = self.givens_property(d)
            self.c["givens_property"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.min_one_prob(d)
            self.c["min_one_prob"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.given_group(d)
            self.c["given_group"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.dyn_set_group(d)
            self.c["dyn_set_group"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.take_group(d)
            self.c["take_group"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.take_args_diff(d)
            self.c["take_args_diff"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

            c = self.parenth(d)
            self.c["parenth"] += c
            self.c_f[os.path.splitext(os.path.basename(d[0]))[0]] += c

    def arity(self, d):
        """
        if arity of predicted predicate is smaller than the minimum arity of the predicate, score
        - incomplete: no embedded predicates
        """
        f = d[0]
        for p, a in iter_p_a(d[1]):
            if p in pred_arity:
                if len(re.findall(",", a)) + 1 < pred_arity[p]:
                    return True
        return False

    def N_is_integer(self, d, n_symb=False):
        """
        check those predicates which should have N at a certain position
        """
        def is_int(w):
            if n_symb:
                return w.lower().startswith("n")
            else:
                try:
                    _ = int(w)
                    return True
                except ValueError:
                    return False

        c = Counter()
        f = d[0]
        for p, a in iter_p_a(d[1]):
            if p in {"size", "take", "take_wr"}:
                n = a.split(",")[-1]
                if not is_int(n):
                    c[p] = 1
            #elif p in {"exactly", "atleast", "atmost", "more_than", "less_than", "nth"}:
            #    n = a.split(",")[0]
            #    if not n.lower().startswith("n"):
            #        c[p] = 1
        if n_symb:
            h = re.findall("(exactly|atleast|atmost|more_than|less_than|nth)\((.)\d,.*,.*\)", "\n".join(d[1]))
            for g in h:
                if g[1] != "n":
                    c[g[0]] = 1
            return c
        else:
            h = re.findall("(exactly|atleast|atmost|more_than|less_than|nth)\(([^,]+),.*,.*\)", "\n".join(d[1]))
            for g in h:
                if "rel" in g[1]:
                    continue
                try:
                    _ = int(g[1])
                except ValueError:
                    c[g[0]] = 1
            return c

    def all_attr_vals_diff(self, d):
        f = d[0]
        for p, a in iter_p_a(d[1]):
            h = re.findall("\[.*\]", a)
            if len(h) == 1:
                l = h.pop()[1:-1].split(",")
                if len(set(l)) < len(l):
                    return True
        return False

    def repeated_statements(self, d):
        f = d[0]
        return len({s for s in d[1]}) < len(d[1])

    def repeated_property_vals(self, d):
        f = d[0]
        vals = []
        for p, a in iter_p_a(d[1]):
            if p == "property":
                h = re.findall("\[.*\]", a)
                if not h:
                    continue
                assert len(h) == 1
                vals.append(h.pop())
        return len(set(vals)) < len(vals)

    def givens_for_group(self, d):
        f = d[0]
        pas = {(p, a) for p, a in iter_p_a(d[1])}
        g_as = {a for p, a in pas if p == "group" and a.startswith("l")}
        for g_a in g_as:
            if re.findall(f"size\({g_a},", "\n".join(d[1])):
                continue
            h = re.findall(f"given\(exactly\(.*,{g_a},.*", "\n".join(d[1]))  # approximate...
            if len(h) < 2:
                return True
        return False

    def givens_property(self, d):
        """
        in given(exactly(*,*,P)), P must be used in exactly one property
        if P is and(P_1, P_2), they need to be mentioned in different properties (?) TODO
        """
        f = d[0]
        pas = {(p, a) for p, a in iter_p_a(d[1])}
        g_as = {a.rsplit(",", 1)[-1].rstrip(")") for p, a in pas if p == "given" and a.startswith("exactly")}
        for g_a in g_as:
            h = re.findall(f"property\(.*{g_a}", "\n".join(d[1]))
            if len(h) != 1:
                return True
        return False

    def min_one_prob(self, d):
        f = d[0]
        for p, a in iter_p_a(d[1]):
            if p == "probability":
                return False
        return True

    def given_group(self, d):
        """
        given(exactly(rel(*,G,*),G,*)) --> group(G)
        """
        f = d[0]
        pas = {(p, a) for p, a in iter_p_a(d[1])}
        g_as = {a for p, a in pas if p == "given" and a.startswith("exactly(rel(")}
        for g_a in g_as:
            # "given(exactly(rel('/'(n0,n1),l0),l0,n3)"
            # "given(exactly(rel(n0,l1),l0,l3)"
            # "given(exactly(rel(n0,l0,l1),l0,and(l2,l1))"
            # "given(exactly(rel('/'(n0,n1),l0,l1),l0,and(l2,l1))"
            h = re.findall("rel\(.*\(.*,.*\),(l.)\),(l.),.*\)", g_a) +\
            re.findall("rel\(.*,(l.)\),(l.),.*\)", g_a) +\
            re.findall("rel\(.*,(l.),.*\),(l.),and\(.*,.*\)\)", g_a) + \
            re.findall("rel\(.*\(.*,.*\),(l.),.*\),(l.),and\(.*,.*\)\)", g_a)
            for g in h:
                g1, g2 = g
                if g1 != g2:
                    continue
                h_g = re.findall(f"group\(.*{g1}", "\n".join(d[1]))
                if len(h_g) == 0:
                    return True
        return False

    def dyn_set_group(self, d):
        """
        all(D,*), some(D,*), exactly(*,D,*), ... -> take/take_wr(*, D, *)
        """
        h = re.findall("(all|some|none|all_same|all_diff|aggcmp)\((l.)", "\n".join(d[1])) + \
        re.findall("(atleast|atmost|more_than|less_than|nth)\(..,(l.),", "\n".join(d[1]))
        for p, g in h:
            h_g = re.findall(f"(take|take_wr)\(.*,{g},.*\)", "\n".join(d[1]))
            if len(h_g) == 0:
                return True
        return False

    def take_group(self, d):
        """
        take/take_wr(M,*,*) --> group(M)
        """
        h = re.findall("(take|take_wr)\((l.)", "\n".join(d[1]))
        for p, g in h:
            h_g = re.findall(f"group\(.*{g}.*\)", "\n".join(d[1]))
            if len(h_g) == 0:
                 return True
        return False

    def take_args_diff(self, d):
        """
        take/take_wr(M,D,*) --> M!=D
        """
        h = re.findall("(take|take_wr)\((l.),(l.),", "\n".join(d[1]))
        for p, g1, g2 in h:
            if g1 == g2:
                return True
        return False

    def parenth(self, d):
        c = 0
        for i in d[1]:
            if len(re.findall("\(", i)) != len(re.findall("\)", i)):
                c += 1

        return c
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    #arg_parser.add_argument("--data-dir", type=str, default="/home/suster/Apps/out/log_w20190906_001042_206510/",
    #arg_parser.add_argument("--data-dir", type=str, default="/home/suster/Apps/out/log_w20191015_161430_083999/",
    #arg_parser.add_argument("--data-dir", type=str, default="/nfshome/suster/Apps/out/log_w20191025_143451_929403_orig/", help="path to folder to be analyzedd")
    arg_parser.add_argument("--data-dir", type=str, default="/nfshome/suster/Apps/out/log_w20191108_093708_793772/")
    args = arg_parser.parse_args()

    fs = get_file_list(args.data_dir, [".pl_p0"])  # prediction files
    data = []
    for f in fs:
        with open(f) as fh:
            data.append((f, fh.read().split()))
    con = ConstraintStats(data)
    con.get_all()
    print(con)
