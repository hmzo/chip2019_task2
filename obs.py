import pandas as pd
import os
from pathlib import Path

OUTPUT = Path("./final_output")
ROOT = Path("./final_data")


def obs(f1, f2):
    a = pd.read_csv(OUTPUT / f1)
    b = pd.read_csv(OUTPUT / f2)
    c = pd.read_csv("./final_data/test_final.csv")

    a.columns = ['id', "label1"]
    b.columns = ['id', "label2"]

    abc = pd.concat([a, b, c], axis=1).drop(['id', 'category'], axis=1)
    return abc


def corr(fs):
    ps = []
    count = 0
    for f in fs:
        model_name = f[:-23]
        tmp = pd.read_csv(ROOT / f)
        tmp.columns = ['id', model_name, 'label']
        ps.append(tmp)
        count += 1
    c = pd.concat(ps, axis=1).drop(['id', 'label'], axis=1)
    return c.corr()