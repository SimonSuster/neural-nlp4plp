from sklearn.metrics import mean_absolute_error

with open("nlp4plp.txt") as fh, open("test_ids") as fh_ids:
    ids = {l.strip() for l in fh_ids.readlines()}
    assert len(ids) == 214

    n_test_preds = 0
    n_corr_test_preds = 0
    preds = []
    golds = []
    n_unsolved = 0
    for l in fh:
        if l.strip():
            id, cat, _ = l.split(" ", 2)
            if id in ids:
                n_test_preds += 1
                if cat == "CORRECT":
                    n_corr_test_preds += 1
                if cat != "ERROR":
                    pred, gold = _.split(" ")
                    if pred != "None" and gold != "None":
                        preds.append(float(pred))
                        golds.append(float(gold))
                if cat in {"SKIPPED", "ERROR", "NO_OUTPUT", "TIMEOUT", "UNKNOWN"}:
                    n_unsolved += 1
    print(f"acc:{(n_corr_test_preds/n_test_preds):.3f}")
    print(n_corr_test_preds)

    print(f"mae: {mean_absolute_error(golds, preds)}")
    print(f"n unsolved: {n_unsolved}")

