#! -*- coding:utf-8 -*-
# 通过R-Drop增强模型的泛化性能
# 数据集：TNEWS 短文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/8496

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from keras.losses import kullback_leibler_divergence as kld
from tqdm import tqdm

labels = [
    "100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112",
    "113", "114", "115", "116"
]
num_classes = len(labels)
maxlen = 128
batch_size = 32

# BERT base
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, labels.index(label)))
    return D


# 加载数据集
train_data = load_data(
    '/root/CLUE-master/baselines/CLUEdataset/tnews/train.json'
)
valid_data = load_data(
    '/root/CLUE-master/baselines/CLUEdataset/tnews/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


model.compile(
    loss=crossentropy_with_rdrop,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': labels[label]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=50,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/tnews/test.json', 'tnews_predict.json')
