import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

ctx = mx.gpu()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps


def load_data(model_name, batch_size=128, train_size=0.8):
    features = nd.load('features_train_%s.nd' % model_name)[0]
    labels = nd.load('labels.nd')[0]

    n_train = int(features.shape[0] * train_size)

    X_train = features[:n_train]
    y_train = labels[:n_train]

    X_val = features[n_train:]
    y_val = labels[n_train:]

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    dataset_val = gluon.data.ArrayDataset(X_val, y_val)

    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size)
    data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

    return data_iter_train, data_iter_val

def build_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(120))

    net.initialize(ctx=ctx)
    return net


def train_model(model_name):
    epochs = 50
    batch_size = 128

    data_iter_train, data_iter_val = load_data(model_name, batch_size)
    net = build_model()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})

    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        steps = len(data_iter_train)
        for data, label in data_iter_train:
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)

            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)

            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        val_loss, val_acc = evaluate(net, data_iter_val)

    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%% Model: %s" % (
        epoch + 1, train_loss / steps, train_acc / steps * 100, val_loss, val_acc * 100, model_name))

    return val_loss

from mxnet.gluon.model_zoo.model_store import _model_sha1

losses = []

for model_name in sorted(_model_sha1.keys()):
    val_loss = train_model(model_name)
    losses.append((model_name, val_loss))

df = pd.DataFrame(losses, columns=['model', 'val_loss'])
df = df.sort_values('val_loss')
df.head()

df.to_csv('models.csv', index=None)
df = pd.read_csv('models.csv')


for i, (model_name, val_loss) in df.iterrows():
    print '%s | %s' % (model_name, val_loss)


def load_models_data(model_names, batch_size=128, train_size=0.8):
    features = [nd.load('features_train_%s.nd' % model_name)[0] for model_name in model_names]
    features = nd.concat(*features, dim=1)
    labels = nd.load('labels.nd')[0]

    n_train = int(features.shape[0] * train_size)

    X_train = features[:n_train]
    y_train = labels[:n_train]

    X_val = features[n_train:]
    y_val = labels[n_train:]

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    dataset_val = gluon.data.ArrayDataset(X_val, y_val)

    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

    return data_iter_train, data_iter_val

df.head(10)
net = build_model()

model_names = ['inceptionv3', 'resnet152_v1']
data_iter_train, data_iter_val = load_models_data(model_names, batch_size=batch_size)

epochs = 100
batch_size = 128
lr_sch = mx.lr_scheduler.FactorScheduler(step=400, factor=0.9)
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'learning_rate': 1e-4, 'wd': 1e-5, 'lr_scheduler': lr_sch})

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    val_loss, val_acc = evaluate(net, data_iter_val)

    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
        epoch+1, train_loss/steps, train_acc/steps*100, val_loss, val_acc*100))

features_test = [nd.load('features_test_%s.nd' % model_name)[0] for model_name in model_names]
features_test = nd.concat(*features_test, dim=1)

output = nd.softmax(net(features_test.as_in_context(ctx))).asnumpy()

df_pred = pd.read_csv('sample_submission.csv')

for i, c in enumerate(df_pred.columns[1:]):
    df_pred[c] = output[:,i]

df_pred.to_csv('pred.csv', index=None)

zip(np.argmax(pd.read_csv('pred_0.28.csv').values[:,1:], axis=-1), np.argmax(df_pred.values[:,1:], axis=-1))[:10]

