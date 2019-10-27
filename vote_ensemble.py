import pandas as pd
import os
import numpy as np
from collections import OrderedDict

root = "./output/"
old_score_rlt = dict()

singles = ["ensemble_base_bert_by_sklearn_lr_stacking_LB_0.8726.csv",
           "ensemble_base_esim_dot_by_sklearn_lr_stacking_LB_0.8006.csv",
           "ensemble_bert&category_transfer_v2_by_sklearn_lr_stacking_0.8740.csv",
           "ensemble_esimedbert_3_by_sklearn_lr_stacking_0.8743.csv",
           "ensemble_robert_by_sklearn_lr_stacking_0.8792.csv"]

for name in singles:
    if name[:-4].split('_')[-1].startswith('0.'):
        score = float(name[:-4].split('_')[-1])
        rlt = pd.read_csv(root + name)
        old_score_rlt[score] = rlt

score_rlt = OrderedDict()
for key in sorted(old_score_rlt.keys(), reverse=True):
    score_rlt[key] = old_score_rlt[key]

count = 0
for score, rlt in score_rlt.items():
    if count == 0:
        rlt.columns = ['id', 'label_%.4f' % score]
        merged = rlt
    else:
        rlt.columns = ['id', 'label_%.4f' % score]
        merged = pd.merge(left=merged, right=rlt, on=['id'])
    count += 1
    if count == len(score_rlt):
        break

columns = merged.columns
merged['vote'] = 3 * merged[columns[1]] + merged[columns[2]] + merged[columns[3]] + merged[columns[4]] + merged[columns[5]]
part_1 = merged[merged['vote'] >= 4]
part_1['label'] = 1
part_0 = merged[merged['vote'] < 4]
part_0['label'] = 0
merged = pd.concat([part_1, part_0]).sort_values(by=['id']).reset_index(drop=True)
output = merged[['id', 'label']]
output.to_csv(root + "vote_by_bert_esim_bert&category_esimedbert_robert.csv", index=False)
