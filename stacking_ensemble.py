import pandas as pd
import numpy as np
from pathlib import Path
import os

from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import Callback
import keras.backend as K

from sklearn.metrics import f1_score, precision_score, recall_score

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

latent_dim = 8
mode = None
binary_classifier_threshold = 0.5
np.random.seed(123)
ROOT = Path("./data")
OUTPUT = Path("./output")
MODEL_SAVED = Path("./model_saved")


data = []
for index, row in pd.read_csv(
        ROOT / "base_bert_stacking_new_train.csv").iterrows():
    data.append((row["probs"], row["label"]))

test_data = []
for index, row in pd.read_csv(
        ROOT / "base_bert_stacking_new_test.csv").iterrows():
    test_data.append((row["probs"], row["id"]))


class DataGenerator:
    def __init__(self, raw_data, batch_size=32, test=False):
        self.data = raw_data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.test = test
        if not test:
            logger.info("__Shuffle the dataset__")
            self.idxs = [x for x in range(len(self.data))]
            np.random.shuffle(self.idxs)
            self.all_labels = self._get_all_label_as_ndarray()
            self.all_labels = self.all_labels[self.idxs]
        else:
            self.ID = self._get_all_id_as_ndarray()

    def __len__(self):
        return self.steps

    def _get_all_label_as_ndarray(self):
        if self.test:
            return None
        all_labels = []
        for x in self.data:
            all_labels.append(x[-1])
        return np.array(all_labels, dtype=np.float32)

    def _get_all_id_as_ndarray(self):
        if not self.test:
            return None
        all_ids = []
        for x in self.data:
            all_ids.append(x[-1])
        return np.array(all_ids, dtype=np.int32)

    def iterator(self):
        while True:
            if not self.test:
                X, Y = [], []
                for i in self.idxs:
                    d = self.data[i]
                    x, y = d[0], d[1]  # x can be a list/ndarray

                    X.append([x])
                    Y.append([y])
                    if len(X) == self.batch_size or i == self.idxs[-1]:
                        yield np.array(X, np.float32), np.array(Y, np.float32)
                        X, Y = [], []
            else:
                X, Y = [], []
                for i in range(len(self.data)):
                    d = self.data[i]
                    x, y = d[0], d[1]  # x can be a list/ndarray

                    X.append([x])
                    Y.append([y])
                    if len(X) == self.batch_size or i == len(self.data) - 1:
                        yield np.array(X, np.float32), None
                        X, Y = [], []


class Evaluator(Callback):
    def __init__(self, model_name, valid_ds):
        super(Evaluator, self).__init__()
        self._best_f1 = 0.
        self._model_saved = MODEL_SAVED / model_name
        self.valid_ds = valid_ds
        if not os.path.exists(self._model_saved):
            os.makedirs(self._model_saved)

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (
            np.asarray(
                self.model.predict_generator(
                    self.valid_ds.iterator(),
                    steps=len(self.valid_ds))))
        val_predict = np.squeeze(val_predict)
        for i in range(len(val_predict)):
            if val_predict[i] >= binary_classifier_threshold:
                val_predict[i] = 1.0
            else:
                val_predict[i] = 0.0
        val_targ = self.valid_ds.all_labels
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        print("One epoch ended, evaluator hook is called")
        print("-val_f1_measure: ", round(_val_f1, 4),
              "\t-val_p_measure: ", round(_val_precision, 4),
              "\t-val_r_measure: ", round(_val_recall, 4))
        if _val_f1 > self._best_f1:

            file_names = os.listdir(self._model_saved)
            for fn in file_names:
                if "mode_%s" % str(mode) in fn:
                    logger.info(
                        "Delete %s from %s" %
                        ((self._model_saved / fn).name, self._model_saved))
                    os.remove(self._model_saved / fn)

            logger.info(
                "Write %s into %s" %
                ("mode_%s_F1_%s.weights" %
                 (str(mode), str(
                     round(
                         _val_f1, 4))), self._model_saved))
            self.model.save_weights(self._model_saved /
                                    ("mode_%s_F1_%s.weights" %
                                     (str(mode), str(round(_val_f1, 4)))))
            self._best_f1 = _val_f1


def create_lr_model():
    x_in = Input(shape=(None,))
    p = Dense(units=1, activation='sigmoid')(x_in)
    model = Model(x_in, p)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def create_mlp_model():
    x_in = Input(shape=(1,))
    x_act = Dense(units=latent_dim, activation='relu')(x_in)
    p = Dense(units=1, activation='sigmoid')(x_act)
    model = Model(x_in, p)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


def stacking(weights_root_path, output_name):
    mode_test_ds = DataGenerator(test_data, batch_size=32, test=True)
    test_ids = mode_test_ds.ID
    test_probs = []
    for weight in os.listdir(weights_root_path):
        model = create_mlp_model()  # todo: be easy to modify it
        model.load_weights(weights_root_path / weight)
        test_probs.append(
            np.squeeze(
                model.predict_generator(
                    mode_test_ds.iterator(),
                    steps=len(mode_test_ds))))
        K.clear_session()

    test_out = pd.DataFrame(
        {"id": test_ids, "probs": np.mean(test_probs, axis=0)})

    preds = []
    ids = []
    for i, r in test_out.iterrows():
        if r['probs'] > binary_classifier_threshold:
            preds.append(1)
        else:
            preds.append(0)
        ids.append(r['id'])

    pd.DataFrame({"id": np.array(ids, np.int32), "label": preds}).to_csv(
        OUTPUT / output_name, index=False)


if __name__ == "__main__":
    random_order = [x for x in range(len(data))]
    np.random.shuffle(random_order)

    for mode in range(10):
        train_data = [
            data[j] for i,
            j in enumerate(random_order) if i %
            10 != mode]
        valid_data = [
            data[j] for i,
            j in enumerate(random_order) if i %
            10 == mode]

        _train_ds = DataGenerator(train_data)
        _valid_ds = DataGenerator(valid_data)
        _test_ds = DataGenerator(test_data, test=True)

        _model = create_mlp_model()
        evaluator = Evaluator("base_bert_stacking_mlp", _valid_ds)
        _model.fit_generator(_train_ds.iterator(),
                             steps_per_epoch=len(_train_ds),
                             epochs=20,
                             validation_data=_valid_ds.iterator(),
                             validation_steps=len(_valid_ds),
                             callbacks=[evaluator])
        K.clear_session()

    stacking(
        MODEL_SAVED / "base_bert_stacking_mlp",
        "ensemble_10_fold_valid_by_mlp.csv")
