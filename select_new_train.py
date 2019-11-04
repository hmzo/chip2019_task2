import pandas as pd
import numpy as np
import re
from collections import OrderedDict
import os

ROOT = "./final_output/"


def select_new_train(root: str, top_k: int = 0, threshold: float = 1.):
    """
    :param top_k: 采用多少个模型进行投票筛选
    :param threshold: 投票结果的阈值
    :param root:
    """
    old_score_rlt = dict()

    for name in os.listdir(root):
        if name[:-4].split('_')[-1].startswith('0.'):
            score = float(name[:-4].split('_')[-1])
            rlt = pd.read_csv(root + name)
            old_score_rlt[score] = rlt

    score_rlt = OrderedDict()
    for key in sorted(old_score_rlt.keys(), reverse=True):
        score_rlt[key] = old_score_rlt[key]

    if top_k == 0:
        top_k = len(score_rlt)

    count = 0
    for score, rlt in score_rlt.items():
        if count == 0:
            rlt.columns = ['id', 'label_%f' % score]
            merged = rlt
        else:
            rlt.columns = ['id', 'label_%f' % score]
            merged = pd.merge(left=merged, right=rlt, on=['id'])
        count += 1
        if count == top_k:
            break

    merged['label_sum'] = np.sum(merged.values[:, 1:], axis=1)
    dev_id = pd.read_csv("./data/dev_id.csv")
    merged = pd.merge(left=dev_id, right=merged, on=['id'])

    selected_0 = merged[merged['label_sum'] <= (1 - threshold) * top_k]
    selected_1 = merged[merged['label_sum'] >= threshold * top_k]

    drop_columns = set([x[0] if len(x) == 1 else None for x in map(
        lambda x: re.compile('label.+').findall(x), selected_0.columns)])
    drop_columns.remove(None)

    selected_0 = selected_0.drop(drop_columns, axis=1)
    selected_0['label'] = 0
    selected_1 = selected_1.drop(drop_columns, axis=1)
    selected_1['label'] = 1

    new_train = pd.concat([selected_0, selected_1]).sort_values(by=['id']).reset_index(drop=True)
    return new_train
