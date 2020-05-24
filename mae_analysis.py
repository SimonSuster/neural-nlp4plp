import json
from statistics import mean, median
import numpy as np

# main point of analysis: ideal answer not often present, but answer with low MAE is!



with open('/home/fivez/math_problems/dev_preds_topk50.json', 'r') as f:
    preds = json.load(f)


def performance_ceiling(preds):

    filtered_preds = []
    for identifier, maes in preds.items():
        maes = [x for x in maes if x]
        if maes:
            maes = [round(x, 3) for x in maes]
            filtered_preds.append(maes)
    print(len(filtered_preds), len(preds))

    # show true performance
    true_mae = mean(x[0] for x in filtered_preds)
    true_median_ae = median(x[0] for x in filtered_preds)
    true_accuracies = {t: sum(1 for x in filtered_preds if x[0] <= t) / len(filtered_preds) for t in [0.1, 0.2, 0.3]}
    print(f'True accuracies: {true_accuracies}')

    # show performance ceiling
    ceiling_mae = mean(min(x) for x in filtered_preds)
    ceiling_accuracies = {t: sum(1 for x in filtered_preds if min(x) <= t) / len(filtered_preds) for t in [0.1, 0.2, 0.3]}

    # show proportion of perfect answer present
    print(f'True MAE: {true_mae}')
    print(f'True Median AE: {true_median_ae}')

    print(f'Highest possible MAE: {ceiling_mae}')
    print(f'Highest possible accuracies: {ceiling_accuracies}')

    # next question: if we take first rank of every list under certain threshold
    # this reads: if we have an estimator which always has at most 0.3 MAE,
    # and use it to filter out the predictions which are possibly more than 0.6 removed from this estimator
    # we get as result the "realistic MAE"
    robustness = 0.35
    threshold = 0.6
    thresholded = []
    for pr in filtered_preds:
        selected = [x for x in pr if x + robustness <= threshold]
        if selected:
            new_value = selected[0]
        else:
            new_value = pr[0]
        thresholded.append(new_value)
    achievable_mae = mean(thresholded)
    achievable_median_ae = median(thresholded)
    print(f'Achievable Mean AE: {achievable_mae}')
    print(f'Achievable Median AE: {achievable_median_ae}')
    achievable_accuracies = {t: sum(1 for x in thresholded if x <= t) / len(thresholded) for t in [0.1, 0.2, 0.3]}
    print(f'Achievable accuracies: {achievable_accuracies}')

    return thresholded


if __name__ == "__main__":

    with open('/home/fivez/math_problems/dev_preds_topk50.json', 'r') as f:
        preds = json.load(f)
    performance_ceiling(preds)
