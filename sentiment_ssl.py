#! -*- coding:utf-8 -*-
# 通过R-Drop进行半监督学习
# use_rdrop=True 比 use_rdrop=False 约有1.5%的提升
# 数据集：情感分析数据集
# 博客：https://kexue.fm/archives/7466

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from keras.utils import to_categorical
from tqdm import tqdm
from keras.losses import kullback_leibler_divergence as kld
import warnings
warnings.filterwarnings("ignore")

# 配置信息
num_classes = 2
maxlen = 128
batch_size = 32
train_frac = 0.01  # 标注数据的比例
use_rdrop = True  # 可以比较True/False的效果

# BERT base
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/sentiment/sentiment.train.data')
valid_data = load_data('datasets/sentiment/sentiment.valid.data')
test_data = load_data('datasets/sentiment/sentiment.test.data')

# 模拟标注和非标注数据
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 0) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = to_categorical(batch_labels, num_classes)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class data_generator_rdrop(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = to_categorical(batch_labels, num_classes)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

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

# 用于正常训练的模型
model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['categorical_accuracy'],
)


def kld_rdrop(y_true, y_pred):
    """无监督部分只需训练KL散度项
    """
    loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return K.mean(loss)


# 用于R-Drop训练的模型
model_rdrop = keras.models.Model(bert.model.input, output)
model_rdrop.compile(
    loss=kld_rdrop,
    optimizer=Adam(1e-5),
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.
        self.data = data_generator_rdrop(unlabeled_data, batch_size).forfit()

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

    def on_batch_end(self, batch, logs=None):
        if use_rdrop:
            dx, dy = next(self.data)
            model_rdrop.train_on_batch(dx, dy)


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=30,
        epochs=100,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
