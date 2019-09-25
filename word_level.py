from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
import os
import logging
import jieba
import thulac
import pkuseg
from typing import List, Tuple

import tensorflow as tf

import keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path("./data")
MODEL_SAVED = Path("./model_saved")
OUTPUT = Path("./output")
WORD_VEC_PATH = Path()

random_order_2000 = np.fromfile("./random_order_2000.npy", dtype=np.int32)
random_order_10000 = np.fromfile("./random_order_10000.npy", dtype=np.int32)
random_order_2500 = np.fromfile("./random_order_2500.npy", dtype=np.int32)
random_order_18000 = np.fromfile("./random_order_18000.npy", dtype=np.int32)

if not os.path.exists(MODEL_SAVED):
    os.makedirs(MODEL_SAVED)

mode = None
freq = 1
learning_rate = 4e-4
min_learning_rate = 1e-4
binary_classifier_threshold = 0.5
tokenize = jieba.cut
# tokenize = pkuseg.pkuseg(model_name='medicine').cut
# tokenize = thulac.thulac().cut

categories = ["aids", "breast_cancer", "diabetes", "hepatitis", "hypertension"]

token_count = dict()
aids_data = []
breast_cancer_data = []
diabetes_data = []
hepatitis_data = []
hypertension_data = []
for index, row in pd.read_csv(ROOT / "train_id.csv").iterrows():
    tokenized_q1 = [x[0] for x in tokenize(row['question1'])]
    tokenized_q2 = [x[0] for x in tokenize(row['question2'])]
    if row["category"] == "aids":
        aids_data.append((tokenized_q1,
                          tokenized_q2,
                          row['category'],
                          row['label'],
                          row["id"]))

    elif row["category"] == "breast_cancer":
        breast_cancer_data.append((tokenized_q1,
                                   tokenized_q2,
                                   row['category'],
                                   row['label'],
                                   row["id"]))
    elif row["category"] == "diabetes":
        diabetes_data.append((tokenized_q1,
                              tokenized_q2,
                              row['category'],
                              row['label'],
                              row["id"]))
    elif row["category"] == "hepatitis":
        hepatitis_data.append((tokenized_q1,
                               tokenized_q2,
                               row['category'],
                               row['label'],
                               row["id"]))
    elif row["category"] == "hypertension":
        hypertension_data.append((tokenized_q1,
                                  tokenized_q2,
                                  row['category'],
                                  row['label'],
                                  row["id"]))
    for token in tokenized_q1:
        token_count[token] = token_count.get(token, 0) + 1
    for token in tokenized_q2:
        token_count[token] = token_count.get(token, 0) + 1

test_data = []
for index, row in pd.read_csv(ROOT / "dev_id.csv").iterrows():
    tokenized_q1 = [x for x in tokenize(row['question1'])]
    tokenized_q2 = [x for x in tokenize(row['question2'])]
    test_data.append(
        (tokenized_q1,
         tokenized_q2,
         row['category'],
         row['id']))
    for token in tokenized_q1:
        token_count[token] = token_count.get(token, 0) + 1
    for token in tokenized_q2:
        token_count[token] = token_count.get(token, 0) + 1

token_index = {"<pad>": 0, "<unk>": 1}
for token, count in token_count.items():
    if count > freq:
        token_index[token] = len(token_index)
index_token = dict((i, x) for x, i in token_index.items())

# w2v embedding (tencent)
conn = sqlite3.connect('./data/w2v.db')
c = conn.cursor()
oov_count = 0
embedding = np.zeros(shape=(len(token_index), 200), dtype=np.float32)
f = open("./oov.txt", 'w')
for word, index in token_index.items():
    try:
        vec_str = c.execute(
            "SELECT vector FROM tencent_w2v WHERE word=?", (word,)).fetchall()[0][0]
        embedding[index] = np.fromstring(vec_str, np.float32)
    except IndexError:
        oov_count += 1
        f.write(word + '\n')
        embedding[index] = np.random.uniform(-1, 1, (200,)).astype(np.float32)
        logger.debug("%s is not included in this database." % word)
        continue
embedding[0] = np.zeros((200,), np.float32)
logger.info("%d words is out of vocabulary." % oov_count)
conn.commit()
conn.close()
f.close()


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
        """
        data format:
            train or valid: [token_list_of q1, token_list_of_q2, category, label]
            test:           [token_list_of q1, token_list_of_q2, category, id   ]
        """
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
                    x1, x2 = [token_index.get(x, 1) for x in d[0]], [token_index.get(x, 1) for x in d[1]]
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
                    x1, x2 = [token_index.get(x, 1) for x in d[0]], [token_index.get(x, 1) for x in d[1]]
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


class CoAttentionAndCombine(Layer):
    def __init__(self, atype):
        super(CoAttentionAndCombine, self).__init__()
        self.atype = atype

    def build(self, input_shape):
        # Used purely for shape validation.
        if len(input_shape) != 4:
            raise ValueError(
                'A `CoAttentionAndCombine` layer should be called '
                'on a list of 4 inputs')
        if all([shape is None for shape in input_shape]):
            return
        inputs_shapes = [list(shape)
                         for shape in input_shape]  # (x1, m1, x2, m2)
        if self.atype == 'bi_linear':
            self.bi_linear_w = self.add_weight(name='bi_linear_w',
                                               initializer='random_normal',
                                               shape=(inputs_shapes[0][-1], inputs_shapes[0][-1]),
                                               trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        x1 = inputs[0]
        m1 = tf.expand_dims(inputs[1], axis=2)  # (batch_size, seq_len1, 1)
        x2 = inputs[2]
        m2 = tf.expand_dims(inputs[3], axis=1)  # (batch_size, 1, seq_len2)
        mask_similarity_matrix = tf.matmul(
            m1, m2)  # (batch_size, seq_len1, seq_len2)
        mask_similarity_matrix = (mask_similarity_matrix - 1.) * 10000
        similarity_matrix: object = None
        if self.atype == 'dot':
            # (batch_size, seq_len1, seq_len2)
            similarity_matrix = tf.matmul(x1, x2, transpose_b=True)
        elif self.atype == 'bi_linear':
            similarity_matrix = tf.matmul(
                tf.tensordot(
                    x1, self.bi_linear_w, [
                        [2], [0]]), x2, transpose_b=True)
        assert similarity_matrix is not None, "type %s is not in ['dot', 'bi_linear']" % self.atype

        similarity_matrix = tf.add(similarity_matrix, mask_similarity_matrix)
        similarity_matrix_transpose = tf.transpose(
            similarity_matrix, perm=[0, 2, 1])

        alpha1 = tf.nn.softmax(similarity_matrix_transpose, axis=-1)
        alpha2 = tf.nn.softmax(similarity_matrix, axis=-1)

        x1_tilde = tf.matmul(alpha2, x2)
        x2_tilde = tf.matmul(alpha1, x1)

        m1 = tf.concat([x1, x1_tilde, tf.abs(tf.subtract(
            x1, x1_tilde)), tf.multiply(x1, x1_tilde)], axis=-1)
        m2 = tf.concat([x2, x2_tilde, tf.abs(tf.subtract(
            x2, x2_tilde)), tf.multiply(x2, x2_tilde)], axis=-1)
        output: List = [m1, m2]  # ***output must be a List***
        return output

    def compute_output_shape(self, input_shape: List[Tuple]) -> List[Tuple]:
        output_shapes: List[Tuple] = list()  # ***element must be a tuple***
        input_shapes = [input_shape[0], input_shape[2]]  # do not output mask
        for x in input_shapes:
            output_shapes.append((x[0], x[1], 4 * x[2]))
        return output_shapes


def seq_padding(seqs, padding=0):
    lens = [len(x)for x in seqs]
    max_len = max(lens)
    return np.array([np.concatenate([x, [padding] * (max_len - len(x))])
                     if len(x) < max_len else x for x in seqs])


def create_esim_model(atype='bi_linear') -> [Model, Model]:
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    c_in = Input(shape=(None,))
    y_in = Input(shape=(None,))

    mask1 = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(x1_in)
    mask2 = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(x2_in)

    embed_layer = Embedding(input_dim=len(token_index),
                            output_dim=200,
                            mask_zero=False,
                            weights=[embedding],
                            trainable=True)  # keras' mask mechanism is NIUBI,
    # the mask schema: if embedding's mask_zero is True, mask(input) -> layer/model operation -> output
    # ==> keras's mask is hard to support custom layer. so revise something

    x1_embed_dropout = Dropout(0.5)(embed_layer(x1_in))
    x2_embed_dropout = Dropout(0.5)(embed_layer(x2_in))

    input_encoder = Bidirectional(LSTM(units=200,
                                       return_sequences=True,
                                       recurrent_dropout=0.2,
                                       dropout=0.2))
    x1_bar = input_encoder(x1_embed_dropout, mask=mask1)

    x2_bar = input_encoder(x2_embed_dropout, mask=mask2)

    local_inference = CoAttentionAndCombine(atype=atype)

    x1_combined, x2_combined = local_inference([x1_bar, mask1, x2_bar, mask2])

    inference_composition = Bidirectional(LSTM(units=200,
                                               return_sequences=True,
                                               recurrent_dropout=0.2,
                                               dropout=0.2))
    x1_compared = inference_composition(x1_combined, mask=mask1)
    x2_compared = inference_composition(x2_combined, mask=mask2)

    def reduce_mean_with_mask(x, mask):
        dim = K.int_shape(x)[-1]
        seq_len = K.expand_dims(K.sum(mask, 1), 1)  # (batch_size, 1)
        # (batch_size, dim), unknown to the keras' broadcasting
        seq_len_tiled = K.tile(seq_len, [1, dim])
        x_sum = K.sum(x, axis=1)  # (batch_size, dim)
        return x_sum / seq_len_tiled

    def avg_mask1(x): return reduce_mean_with_mask(x, mask1)

    def avg_mask2(x): return reduce_mean_with_mask(x, mask2)

    def max_closure(x): return K.max(x, axis=1)

    avg_op1 = Lambda(avg_mask1)
    avg_op2 = Lambda(avg_mask2)
    max_op = Lambda(max_closure)

    x1_avg = avg_op1(x1_compared)
    x1_max = max_op(x1_compared)
    x2_avg = avg_op2(x2_compared)
    x2_max = max_op(x2_compared)

    x1_rep = Concatenate()([x1_avg, x1_max])
    x2_rep = Concatenate()([x2_avg, x2_max])

    merge_features = Concatenate()([x1_rep, x2_rep])
    hidden = Dropout(0.5)(Dense(200, activation='relu')(merge_features))
    p = Dense(1, activation='sigmoid')(hidden)

    train_model = Model([x1_in, x2_in, c_in, y_in], p)
    model = Model([x1_in, x2_in, c_in], p)

    loss = K.mean(K.binary_crossentropy(target=y_in, output=p))
    train_model.add_loss(loss)

    train_model.compile(optimizer=Adam(learning_rate))
    train_model.summary()

    return train_model, model


def train(train_model, train_ds, valid_ds, model_name):
    evaluator = Evaluator(model_name=model_name, valid_ds=valid_ds, patience=10)

    train_model.fit_generator(train_ds.iterator(),
                              steps_per_epoch=len(train_ds),
                              epochs=100,
                              validation_data=valid_ds.iterator(),
                              validation_steps=len(valid_ds),
                              callbacks=[evaluator])


def predict(model, weights_path, test_ds, valid_result_name, valid_logits_name):
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
        OUTPUT / valid_result_name,
        index=False)
    _valid_logits.to_csv(
        OUTPUT / valid_logits_name,
        index=False)


def gen_stacking_features(weights_root_path, model_name):
    valid_true_labels = []
    valid_probs = []
    valid_ids = []
    mode_test_ds = DataGenerator(
        test_data,
        batch_size=128,
        test=True)
    test_ids = mode_test_ds.ID
    test_probs = []
    for weight in os.listdir(weights_root_path):
        _mode = int(weight.split('_')[1])
        train_model, _ = create_esim_model()  # todo: be easy to modify it
        train_model.load_weights(weights_root_path / weight)

        _valid_data = [aids_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [hypertension_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [hepatitis_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [breast_cancer_data[j] for i, j in enumerate(random_order_2500) if i % 10 == _mode] + \
                      [diabetes_data[j] for i, j in enumerate(random_order_10000) if i % 10 == _mode]

        mode_valid_ds = DataGenerator(
            _valid_data,
            batch_size=128,
            test=False)
        valid_true_labels.append(mode_valid_ds.all_labels)
        valid_ids.append(mode_valid_ds.ID)

        valid_probs.append(
            np.squeeze(
                train_model.predict_generator(
                    mode_valid_ds.iterator(),
                    steps=len(mode_valid_ds))))

        K.clear_session()

        # todo: the next is to deal with the abundant calling
        _, model = create_esim_model()
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
    pd.DataFrame(to_vote_format).to_csv(
        ROOT / (model_name + "_predictions_for_vote.csv"), index=False)

    valid_out = pd.DataFrame({"id": np.concatenate(valid_ids).astype(np.int32),
                              "probs": np.concatenate(valid_probs),
                              "label": np.concatenate(valid_true_labels).astype(np.int32)})
    valid_out.sort_values(by="id", inplace=True)

    test_out = pd.DataFrame(
        {"id": test_ids, "probs": np.mean(test_probs, axis=0)})

    valid_out.to_csv(ROOT / (model_name + "_stacking_new_train.csv"),
                     index=False)  # todo: be easy to modify it
    test_out.to_csv(
        ROOT / (model_name + "_stacking_new_test.csv"), index=False)


if __name__ == "__main__":

    for mode in range(10):
        train_data = [aids_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [hypertension_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] +\
                     [hepatitis_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [breast_cancer_data[j] for i, j in enumerate(random_order_2500) if i % 10 != mode] + \
                     [diabetes_data[j] for i, j in enumerate(random_order_10000) if i % 10 != mode]

        valid_data = [aids_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [hypertension_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [hepatitis_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [breast_cancer_data[j] for i, j in enumerate(random_order_2500) if i % 10 == mode] + \
                     [diabetes_data[j] for i, j in enumerate(random_order_10000) if i % 10 == mode]

        _train_ds = DataGenerator(
            train_data,
            batch_size=128,
            test=False)
        _valid_ds = DataGenerator(
            valid_data,
            batch_size=128,
            test=False)

        _test_ds = DataGenerator(
            test_data,
            batch_size=128,
            test=True)

        _train_model, _model = create_esim_model()

        train(
            train_model=_train_model,
            train_ds=_train_ds,
            valid_ds=_valid_ds,
            model_name="base_esim")
        logger.info("___Reset The Computing Graph___")
        K.clear_session()
    gen_stacking_features(Path(MODEL_SAVED) / "base_esim", "base_esim")
