import pandas as pd
import numpy as np

binary_classifier_threshold = 0.5

test = pd.read_csv(
    "./data/base_bert_stacking_new_test.csv")

preds = []
ids = []
for i, row in test.iterrows():
    if row['probs'] > binary_classifier_threshold:
        preds.append(1)
    else:
        preds.append(0)
    ids.append(row['id'])

pd.DataFrame({"id": np.array(ids, np.int32), "label": preds}).to_csv(
    "./output/ensemble_10_fold_valid_by_experience.csv", index=False)
