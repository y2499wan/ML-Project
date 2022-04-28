# these are the methods in test.py
from typing import Any
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from TransformerNetwork import *
from t2v import *
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from hyperparameters import *


def sliding_window(x, y, window_size):
    xx, yy = [], []
    for i in range(window_size, len(x)):
        xx.append(x[i - window_size:i])
        yy.append(y[i])
    xx, yy = np.array(xx), np.array(yy)
    return xx, yy


def _divide_no_nan(a, b):
    # Auxiliary function to handle divide by 0
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div


def mape_loss(y, y_hat, mask=None):
    if mask is None:
        mask = torch.ones_like(y_hat)

    mask = _divide_no_nan(mask, torch.abs(y))
    mape = torch.abs(y - y_hat) * mask
    mape = torch.mean(mape)
    return mape


def epoch_train(dataloader, model, optimizer, loss_fc, mae):
    model.train()
    batch_size = len(next(iter(dataloader)))
    losses = 0  # mse
    losses1 = 0  # percentage loss

    for x_train_spl, y_train_spl in dataloader:
        # clear the gradients
        optimizer.zero_grad()

        y_pred = model(x_train_spl)
        loss = loss_fc(y_pred, y_train_spl)
        loss1 = mae(y_pred, y_train_spl)

        # calculate gradients
        loss.backward()

        # update weights
        optimizer.step()

        losses += loss.item() / batch_size
        losses1 += loss1.item() / batch_size  # mae

    return losses, losses1


def epoch_test(dataloader, model, loss_fc):
    model.eval()
    batch_size = len(next(iter(dataloader)))
    losses = 0  # mse
    losses1 = 0  # percentage loss
    for x_test, y_test in dataloader:
        # x_test = data
        # y_test = data[:,:,3]
        y_pred = model(x_test)
        # print("y_pred",y_pred.shape)
        # print("y_test",y_test.shape)

        # y_pred = scaler_y.inverse_transform(y_pred.detach().numpy())
        # y_test = scaler_y.inverse_transform(y_test)
        # loss = mean_squared_error(y_pred,y_test)
        # loss1 = np.mean(np.divide(np.abs(y_pred - y_test),y_test))

        loss = loss_fc(y_pred, y_test)
        # loss1 = torch.mean(torch.div(torch.abs(y_pred - y_test),y_test))
        loss1 = mape_loss(y_test, y_pred)

        losses += loss.item() / batch_size
        losses1 += loss1.item() / batch_size  # mae

    return losses, losses1


def run_epoch(epochs, to_output, train_dataset, val_dataset, model, optimizer, loss_fc, mae):
    train = []
    val = []  # float to double -> .astype(np.float32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    for epoch in range(epochs):
        train_mse, train_mae = epoch_train(train_loader, model, optimizer, loss_fc, mae)
        train.append(train_mse)
        val_mse, val_mae = epoch_test(val_loader, model, loss_fc)
        val.append(val_mse)
        str = ('Stock {} Epoch[{}/{}] | train(mse):{:.6f}, mape:{:.6f} | test(mse):{:.6f}, mape:{:.6f}\n'
               .format(stock_name, epoch + 1, epochs, train_mse, train_mae, val_mse, val_mae))
        print(str, end='')
        if epoch == epochs - 1:
            to_output.write(str)
    to_output.close()
    plt.plot(train, label='training_error')
    plt.plot(val, label='validation_error')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.title("Error vs Epoch")
    plt.legend()
    plt.show()
