from pathlib import Path
import numpy as np
import pandas as pd
import codecs
import os
import logging

import tensorflow as tf
import keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.layers import Input, Embedding, Lambda, Dense, Layer
from keras.models import Model
from keras.callbacks import Callback
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, AdamWarmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT = Path("./data")
MODEL_SAVED = Path("./model_saved")

if not os.path.exists(MODEL_SAVED):
    os.makedirs(MODEL_SAVED)


mode = None
max_seq_len = 512
learning_rate = 7e-5
min_learning_rate = 1e-5
binary_classifier_threshold = 0.5
config_path = './bert/bert_config.json'
checkpoint_path = './bert/bert_model.ckpt'
dict_path = './bert/vocab.txt'

random_order_2000 = np.fromfile("./random_order_2000.npy", np.int32)
random_order_2500 = np.fromfile("./random_order_2500.npy", dtype=np.int32)
random_order_10000 = np.fromfile("./random_order_10000.npy", dtype=np.int32)
random_order_18000 = np.fromfile("./random_order_18000.npy", dtype=np.int32)


categories = ["aids", "breast_cancer", "diabetes", "hepatitis", "hypertension"]


token_dict = {}
with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)

aids_data = []
breast_cancer_data = []
diabetes_data = []
hepatitis_data = []
hypertension_data = []
for index, row in pd.read_csv(ROOT / "train_id.csv").iterrows():
    if row["category"] == "aids":
        aids_data.append((row['question1'],
                          row['question2'],
                          row['category'],
                          row['label'],
                          row['id']))

    elif row["category"] == "breast_cancer":
        breast_cancer_data.append((row['question1'],
                                   row['question2'],
                                   row['category'],
                                   row['label'],
                                   row['id']))
    elif row["category"] == "diabetes":
        diabetes_data.append((row['question1'],
                              row['question2'],
                              row['category'],
                              row['label'],
                              row['id']))
    elif row["category"] == "hepatitis":
        hepatitis_data.append((row['question1'],
                               row['question2'],
                               row['category'],
                               row['label'],
                               row['id']))
    elif row["category"] == "hypertension":
        hypertension_data.append((row['question1'],
                                  row['question2'],
                                  row['category'],
                                  row['label'],
                                  row['id']))

new_data = []
for index, row in pd.read_csv(ROOT / "new_train_id.csv").iterrows():
    new_data.append((row['question1'],
                     row['question2'],
                     row['category'],
                     row['label'],
                     row['id']))

test_data = []
for index, row in pd.read_csv(ROOT / "dev_id.csv").iterrows():
    test_data.append(
        (row['question1'],
         row['question2'],
         row['category'],
         row['id']))


class Evaluator(Callback):
    def __init__(self, model_name, valid_ds, patience):
        super(Evaluator, self).__init__()
        self._best_f1 = 0.
        self.passed = 0
        self.best_epoch = -1
        self.epochs = 0
        self.patience = patience
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
            self.best_epoch = self.epochs
            assert isinstance(mode, int), "check mode, must be a integer"
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

        if self.epochs - self.best_epoch > self.patience:
            self.model.stop_training = True
            logger.info("%d epochs have no improvement, earlystoping caused..." % self.patience)
        self.epochs += 1


class DataGenerator:
    def __init__(self, data, batch_size=32, test=False):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.test = test

        if not test:
            logger.info("__Shuffle the dataset__")
            if len(self.data) == 2000:
                self.idxs = random_order_2000
            elif len(self.data) == 18000:
                self.idxs = random_order_18000
            else:
                self.idxs = [x for x in range(len(self.data))]
                np.random.shuffle(self.idxs)
            self.ID = self._get_all_id_as_ndarray()[self.idxs]
            self.all_labels = self._get_all_label_as_ndarray()
            self.all_labels = self.all_labels[self.idxs]
        else:
            self.ID = self._get_all_id_as_ndarray()

    def __len__(self):
        return self.steps

    def iterator(self):
        while True:
            if not self.test:
                X1, X2, C, Y = [], [], [], []
                for i in self.idxs:
                    d = self.data[i]
                    x1, x2 = tokenizer.encode(first=d[0], second=d[1])
                    c, y = d[2], d[3]
                    X1.append(x1)
                    X2.append(x2)
                    C.append([categories.index(c)])
                    Y.append([y])
                    if len(X1) == self.batch_size or i == self.idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        C = seq_padding(C)
                        Y = seq_padding(Y)
                        # todo: too abundant to use the generator
                        yield [X1, X2, C, Y], None
                        X1, X2, C, Y = [], [], [], []
            elif self.test:
                X1, X2, C = [], [], []
                for i in range(len(self.data)):
                    d = self.data[i]
                    x1, x2 = tokenizer.encode(first=d[0], second=d[1])
                    c = d[2]
                    X1.append(x1)
                    X2.append(x2)
                    C.append([categories.index(c)])
                    if len(X1) == self.batch_size or i == len(self.data) - 1:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        C = seq_padding(C)
                        yield [X1, X2, C], None
                        X1, X2, C = [], [], []

    def _get_all_label_as_ndarray(self):
        if self.test:
            return None
        all_labels = []
        for x in self.data:
            all_labels.append(x[-2])
        return np.array(all_labels, dtype=np.float32)

    def _get_all_id_as_ndarray(self):
        all_ids = []
        for x in self.data:
            all_ids.append(x[-1])
        return np.array(all_ids, dtype=np.int32)


def seq_padding(seqs, padding=0):
    lens = [len(x)for x in seqs]
    max_len = max(lens)
    return np.array([np.concatenate([x, [padding] * (max_len - len(x))])
                     if len(x) < max_len else x for x in seqs])


def create_base_bert_model():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    c_in = Input(shape=(None,))
    y_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0, :])(x)  # [CLS]

    p = Dense(1, activation='sigmoid')(x)
    train_model = Model([x1_in, x2_in, c_in, y_in], p)
    model = Model([x1_in, x2_in, c_in], p)

    loss = K.mean(K.binary_crossentropy(target=y_in, output=p))
    train_model.add_loss(loss)

    train_model.compile(optimizer=AdamWarmup(decay_steps=len(_train_ds),
                                             warmup_steps=len(_train_ds),
                                             lr=learning_rate,
                                             min_lr=min_learning_rate))
    train_model.summary()

    return train_model, model


class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if len(input_shape) != 2:
            raise ValueError(
                'A `Attention` layer should be called '
                'on a list of 2 inputs')
        if all([shape is None for shape in input_shape]):
            return
        self.built = True

    def call(self, inputs, **kwargs):
        query, key_value = inputs

        query = K.expand_dims(query, axis=-1)  # (B, D, 1)

        score = tf.matmul(key_value, query)  # (B, S, 1)
        score = K.softmax(score, axis=1)  # (B, S, 1)
        rlt = score * key_value  # (B, S, 1) * (B, S, D) = (B, S, D)
        rlt = tf.reduce_mean(rlt, axis=1)  # (B, D)
        return rlt

    def compute_output_shape(self, input_shape):
        return [input_shape[0]]


def create_bert_concat_category_embedding_model():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    c_in = Input(shape=(None,))
    y_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])

    # -----------bert_concat_category_embedding-------------
    x = Dense(units=50, activation='relu')(x)  # (B, S, D)
    x_cls = Lambda(lambda x: x[:, 0, :])(x)  # [CLS]
    c = Embedding(input_dim=5,
                  output_dim=50,
                  trainable=True)(c_in)
    c = Lambda(lambda c: c[:, 0, :])(c)  # (B, D)

    c = Attention()([c, x])

    x_c_concat = Lambda(K.concatenate)([x_cls, c])
    # ------------------------------------------------------

    p = Dense(1, activation='sigmoid')(x_c_concat)
    train_model = Model([x1_in, x2_in, c_in, y_in], p)
    model = Model([x1_in, x2_in, c_in], p)

    loss = K.mean(K.binary_crossentropy(target=y_in, output=p))
    train_model.add_loss(loss)

    train_model.compile(optimizer=AdamWarmup(decay_steps=len(_train_ds),
                                             warmup_steps=len(_train_ds),
                                             lr=learning_rate,
                                             min_lr=min_learning_rate))
    train_model.summary()

    return train_model, model


def train(train_model, train_ds, valid_ds, model_name):

    evaluator = Evaluator(model_name=model_name, valid_ds=valid_ds, patience=5)

    train_model.fit_generator(train_ds.iterator(),
                              steps_per_epoch=len(train_ds),
                              epochs=30,
                              class_weight="auto",
                              validation_data=valid_ds.iterator(),
                              validation_steps=len(valid_ds),
                              callbacks=[evaluator])


def predict(model, weights_path, test_ds):
    model.load_weights(weights_path)
    probs = model.predict_generator(test_ds.iterator(), steps=len(test_ds))
    probs = np.squeeze(probs)

    preds = []

    for i in range(len(probs)):
        if probs[i] > binary_classifier_threshold:
            preds.append(1)
        else:
            preds.append(0)
    preds = np.array(preds, dtype=np.int32)

    ids = test_ds.ID

    _valid_result = pd.DataFrame({"id": ids, "label": preds})
    _valid_logits = pd.DataFrame({"id": ids, "logits": probs})

    _valid_result.to_csv(
        "./output/simple_bert_output.csv",
        index=False)


def gen_stacking_features(weights_root_path, model_name):
    valid_true_labels = []
    valid_probs = []
    valid_ids = []
    mode_test_ds = DataGenerator(test_data, batch_size=32, test=True)
    test_ids = mode_test_ds.ID
    test_probs = []
    for weight in os.listdir(weights_root_path):
        _mode = int(weight.split('_')[1])
        train_model, _ = create_bert_concat_category_embedding_model()  # todo: be easy to modify it
        train_model.load_weights(weights_root_path / weight)

        _valid_data = [aids_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [hypertension_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [hepatitis_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [breast_cancer_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [diabetes_data[j] for i, j in enumerate(random_order_10000) if i % 10 == _mode]

        mode_valid_ds = DataGenerator(_valid_data, batch_size=32, test=False)
        valid_true_labels.append(mode_valid_ds.all_labels)
        valid_ids.append(mode_valid_ds.ID)

        valid_probs.append(
            np.squeeze(
                train_model.predict_generator(
                    mode_valid_ds.iterator(),
                    steps=len(mode_valid_ds))))

        K.clear_session()

        # todo: the next is dealing with the abundant calling
        _, model = create_bert_concat_category_embedding_model()
        model.load_weights(weights_root_path / weight)
        test_probs.append(
            np.squeeze(
                model.predict_generator(
                    mode_test_ds.iterator(),
                    steps=len(mode_test_ds))))

        K.clear_session()

    to_vote_format = {"id": test_ids}
    for i, each in enumerate(test_probs):
        to_vote_format["label_%s" % str(i)] = np.array(each.round(), np.int32)
    pd.DataFrame(to_vote_format).to_csv(ROOT / (model_name + "_predictions_for_vote.csv"), index=False)

    valid_out = pd.DataFrame({"id": np.concatenate(valid_ids).astype(np.int32),
                              "probs": np.concatenate(valid_probs),
                              "label": np.concatenate(valid_true_labels).astype(np.int32)})
    valid_out.sort_values(by='id', inplace=True)

    test_out = pd.DataFrame(
        {"id": test_ids, "probs": np.mean(test_probs, axis=0)})

    valid_out.to_csv(ROOT / (model_name + "_stacking_new_train.csv"),
                     index=False)
    test_out.to_csv(
        ROOT / (model_name + "_stacking_new_test.csv"), index=False)


if __name__ == "__main__":
    for mode in range(10):
        train_data = [aids_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [hypertension_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [hepatitis_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [breast_cancer_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [diabetes_data[j] for i, j in enumerate(random_order_10000) if i % 10 != mode]

        valid_data = [aids_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [hypertension_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [hepatitis_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [breast_cancer_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [diabetes_data[j] for i, j in enumerate(random_order_10000) if i % 10 == mode]

        inverse_train_data = []
        for x in train_data:
            inverse_train_data.append((x[1], x[0], x[2], x[3], x[4]))

        train_data = train_data + new_data
        _train_ds = DataGenerator(train_data, batch_size=32, test=False)

        _valid_ds = DataGenerator(valid_data, batch_size=32, test=False)

        _test_ds = DataGenerator(test_data, batch_size=32, test=True)

        _train_model, _model = create_bert_concat_category_embedding_model()  # todo: be easy to modify it

        train(
            train_model=_train_model,
            train_ds=_train_ds,
            valid_ds=_valid_ds,
            model_name="base_bert_add_category_transfer_1")
        logger.info("___Reset The Computing Graph___")
        K.clear_session()
    gen_stacking_features(Path(MODEL_SAVED) / "base_bert_add_category_transfer_1", "base_bert_add_category_transfer_1")

    # _, model = create_base_bert_model()
    # test_ds = DataGenerator(test_data, test=True)
    # predict(model, "", test_ds)
