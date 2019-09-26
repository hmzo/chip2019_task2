import pandas as pd
import numpy as np

all_test_pred = pd.read_csv(
    "./data/base_bert_predictions_for_vote.csv")

preds = []
ids = all_test_pred["id"]
for _, row in all_test_pred.iterrows():
    one_count = 0
    zero_count = 0
    for i in range(10):
        n = "label_%s" % str(i)
        if row[n] == 1:
            one_count += 1
        elif row[n] == 0:
            zero_count += 1
    if one_count >= zero_count:
        preds.append(1)
    else:
        preds.append(0)

pd.DataFrame({"id": np.array(ids, np.int32), "label": np.array(preds, np.int32)}).to_csv(
    "./output/ensemble_base_bert_by_vote.csv", index=False)
