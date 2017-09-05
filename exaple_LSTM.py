#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
from numpy.random import *
import six

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L
# cudaがある環境ではコメントアウト解除
from chainer import cuda

# LSTMのネットワーク定義
class LSTM(chainer.Chain):
    def __init__(self, p, n_units, train=True):
        super(LSTM, self).__init__(
            embed=L.EmbedID(p+1, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.Linear(n_units, p+1),
        )

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)
        return y

    def reset_state(self):
        self.l1.reset_state()

# 引数の処理
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# パラメータ設定
p = 5          # 文字列長
n_units = 4    # 隠れ層のユニット数

# 訓練データの準備
# train_data[0]がy、train_data[1]がxの方
# a_1を0にしたので添字がひとつずれている
train_data = np.ndarray((2, p+1), dtype=np.int32)
train_data[0][0] = train_data[0][p] = p
train_data[1][0] = train_data[1][p] = p-1
for i in range(p-1):
    train_data[0][i+1] = i
    train_data[1][i+1] = i

# 訓練データの表示
print(train_data[0])
print(train_data[1])


# モデルの準備
lstm = LSTM(p , n_units)
# このようにすることで分類タスクを簡単にかける
# 詳しくはドキュメントを読むとよい
model = L.Classifier(lstm)
model.compute_accuracy = False
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.2, 0.2, data.shape)

# cuda環境では以下のようにすればよい
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# optimizerの設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# 訓練を行うループ
display = 1000  # 何回ごとに表示するか
total_loss = 0  # 誤差関数の値を入れる変数
for seq in range(100000):
    sequence = train_data[randint(2)] # ランダムにどちらかの文字列を選ぶ
    lstm.reset_state()  # 前の系列の影響がなくなるようにリセット
    for i in six.moves.range(p):
        x = chainer.Variable(xp.asarray([sequence[i]]))   # i文字目を入力に
        t = chainer.Variable(xp.asarray([sequence[i+1]])) # i+1文字目を正解に
        loss = model(x, t)  # lossの計算

        # 出力する時はlossを記憶
        if seq%display==0:
            total_loss += loss.data

        # 最適化の実行
        model.zerograds()
        loss.backward()
        optimizer.update()

    # lossの表示
    if seq%display==0:
        print("sequence:{}, loss:{}".format(seq, total_loss))
        total_loss = 0
    # 10回に1回系列ごとの予測結果と最後の文字の確率分布を表示
    if seq%(display*10)==0:
        for select in six.moves.range(2):
            sequence = train_data[select]
            lstm.reset_state()
            print("prediction: {},".format(sequence[0]), end="")
            for i in six.moves.range(p):
                x = chainer.Variable(xp.asarray([sequence[i]]))
                data = lstm(x).data
                print("{},".format(np.argmax(data)), end="")
            print()
            print("probability: {}".format(data))
