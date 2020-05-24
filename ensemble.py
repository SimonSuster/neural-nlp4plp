import argparse
import os
from collections import Counter

from util import get_file_list


def stats(all_programs):
    n = len(all_programs)
    n_types = 0
    maxes = []
    for prog_id, progs in all_programs.items():
        c = Counter(progs.values())
        n_types += len(c)
        maxes.append(c.most_common(1)[0][1])
    mean_n_types = n_types / n
    return mean_n_types, maxes


def majority_vote(all_programs, method):
    best_programs = {}
    for prog_id, progs in all_programs.items():
        all_p = [p for progs in progs.values() for p in progs]
        c = Counter(all_p)
        #c = Counter(progs.values())
        vote = c.most_common(1)[0][1]
        best_programs[prog_id] = c.most_common(1)[0][0]  # majority vote

    return best_programs


def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--results_dirs", nargs='+', help="ensemble over these")
    arg_parser.add_argument("--method", default="majority_first_best", help="majority_first_best | majority_all")
    args = arg_parser.parse_args()

    if args.results_dirs is None:
        # "[(0.16744186046511628, '20191114_122954_807372'), (0.17209302325581396, '20191114_133329_741729'), (0.1813953488372093, '20191114_142328_283787'), (0.14418604651162792, '20191114_152914_287908'), (0.15813953488372093, '20191114_160239_643445'), (0.15813953488372093, '20191114_165449_081984'), (0.16279069767441862, '20191114_175758_239037'), (0.17674418604651163, '20191114_184101_415459'), (0.1813953488372093, '20191114_193035_425739'), (0.16279069767441862, '20191114_204213_112632')]"
        results_dirs = [
            '/nfshome/suster/Apps/out/log_w20191114_122954_807372',
            '/nfshome/suster/Apps/out/log_w20191114_133329_741729',
            '/nfshome/suster/Apps/out/log_w20191114_142328_283787',
            '/nfshome/suster/Apps/out/log_w20191114_152914_287908',
            '/nfshome/suster/Apps/out/log_w20191114_160239_643445',
            '/nfshome/suster/Apps/out/log_w20191114_165449_081984',
            '/nfshome/suster/Apps/out/log_w20191114_175758_239037',
            '/nfshome/suster/Apps/out/log_w20191114_184101_415459',
            '/nfshome/suster/Apps/out/log_w20191114_193035_425739',
            '/nfshome/suster/Apps/out/log_w20191114_204213_112632'
        ]
    else:
        results_dirs = args.results_dir

    all_programs = {}
    for dir in results_dirs:
        fs = get_file_list(dir, identifiers=[".pl_p"])
        for f in fs:
            prob_id = os.path.splitext(os.path.basename(f))[0]
            with open(f) as fh:
                prog = fh.read()
                if prob_id not in all_programs:
                    all_programs[prob_id] = {}
                if args.method == "majority_first_best":
                    all_programs[prob_id][dir] = prog
                elif args.method == "majority_all":
                    if dir not in all_programs[prob_id]:
                        all_programs[prob_id][dir] = []
                    all_programs[prob_id][dir].append(prog)

    for prog_id in all_programs:
        assert len(all_programs[prog_id]) == len(results_dirs)

    #print(stats(all_programs))

    best_programs = majority_vote(all_programs, method=args.method)

    out_dir = '/home/suster/Apps/out/ensemble/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for prog_id, prog in best_programs.items():
        with open(f"{out_dir}/{prog_id}.pl_p0", "w") as fh:
            fh.write(prog)

if __name__ == '__main__':
    main()

