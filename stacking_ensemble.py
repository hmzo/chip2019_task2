import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression


binary_classifier_threshold = 0.5
ROOT = Path("./final_data")
OUTPUT = Path("./final_output")


train_features_labels = list()
# =========================================================================
# train_features_labels.append(pd.read_csv(ROOT / "robert_stacking_new_train.csv"))
train_features_labels.append(pd.read_csv(ROOT / "robert_transfer_v2_stacking_new_train.csv"))
train_features_labels.append(pd.read_csv(ROOT / "esim_bert_domain_stacking_new_train.csv"))
# train_features_labels.append(pd.read_csv(ROOT / "esim_bert_domain_transfer_v2_stacking_new_train.csv"))
# train_features_labels.append(pd.read_csv(ROOT / "robert_domain_stacking_new_train.csv"))
# train_features_labels.append(pd.read_csv(ROOT / "siamese_esim_bert_domain_stacking_new_train.csv"))
# train_features_labels.append(pd.read_csv(ROOT / "esim_robert_stacking_new_train.csv"))
# train_features_labels.append(pd.read_csv(ROOT / "esim_robert_domain_transfer_v2_stacking_new_train.csv"))
# train_features_labels.append(pd.read_csv(ROOT / "base_bert_add_category_transfer_v2_stacking_new_train.csv"))
# =========================================================================

test_features = list()
# =========================================================================
# test_features.append(pd.read_csv(ROOT / "robert_stacking_new_test.csv"))
test_features.append(pd.read_csv(ROOT / "robert_transfer_v2_stacking_new_test.csv"))
test_features.append(pd.read_csv(ROOT / "esim_bert_domain_stacking_new_test.csv"))
# test_features.append(pd.read_csv(ROOT / "esim_bert_domain_transfer_v2_stacking_new_test.csv"))
# test_features.append(pd.read_csv(ROOT / "robert_domain_stacking_new_test.csv"))
# test_features.append(pd.read_csv(ROOT / "siamese_esim_bert_domain_stacking_new_test.csv"))
# test_features.append(pd.read_csv(ROOT / "esim_robert_stacking_new_test.csv"))
# test_features.append(pd.read_csv(ROOT / "esim_robert_domain_transfer_v2_stacking_new_test.csv"))
# test_features.append(pd.read_csv(ROOT / "base_bert_add_category_transfer_v2_stacking_new_test.csv"))
# =========================================================================

train_features_labels[0].columns = ['id', 'probs_0', 'label']
result = train_features_labels[0]
for i in range(1, len(train_features_labels)):
    train_features_labels[i].columns = ['id', 'probs_%d' % i, 'label']
    result = pd.merge(
        left=result, right=train_features_labels[i], on=[
            "id", "label"])

probs = set([k if ("probs" in k) else None for k in result.keys()])
probs.remove(None)
probs = [x for x in probs]


test_features[0].columns = ['id', 'probs_0']
test_result = test_features[0]
for i in range(1, len(test_features)):
    test_features[i].columns = ['id', 'probs_%d' % i]
    test_result = pd.merge(left=test_result, right=test_features[i], on=["id"])


if __name__ == "__main__":

    X_train, y_train = result[probs].values, result["label"].values
    stacker = LogisticRegression()
    stacker.fit(X_train, y_train)
    print(stacker.coef_)
    X_test = test_result[probs].values
    output = pd.DataFrame({"id": test_result["id"], "label": stacker.predict(X_test).astype(np.int32)})

    output.to_csv(OUTPUT / "第三世界研究室_test_result.csv", index=False)
