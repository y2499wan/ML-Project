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
import uuid


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
    loss_mse = 0  # mse
    loss_mape = 0  # percentage loss

    for x_train_spl, y_train_spl in dataloader:
        # clear the gradients
        optimizer.zero_grad()

        y_pred = model(x_train_spl)
        loss = loss_fc(y_pred, y_train_spl)
        loss1 = mae(y_pred, y_train_spl)

        loss.backward()  # calculate gradients
        optimizer.step()  # update weights

        loss_mse += loss.item() / batch_size
        loss_mape += loss1.item() / batch_size  # mae

    return loss_mse, loss_mape


def epoch_test(dataloader, model, loss_fc):
    model.eval()
    batch_size = len(next(iter(dataloader)))
    loss_mse = 0  # mse
    loss_mape = 0  # percentage loss
    for x_test, y_test in dataloader:
        y_pred = model(x_test)
        cur_mse = loss_fc(y_pred, y_test)
        cur_mape = mape_loss(y_test, y_pred)

        loss_mse += cur_mse.item() / batch_size
        loss_mape += cur_mape.item() / batch_size  # mape

    return loss_mse, loss_mape


def run_epoch(epochs, train_dataset, val_dataset, model, optimizer, loss_fc, mae, to_output):
    # float to double -> .astype(np.float32)
    train_mse, val_mse, train_mape, val_mape = np.zeros((4, epochs))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    filename = 'results/' + str(uuid.uuid1())
    for epoch in range(epochs):
        train_mse[epoch], train_mape[epoch] = epoch_train(train_loader, model, optimizer, loss_fc, mae)
        val_mse[epoch], val_mape[epoch] = epoch_test(val_loader, model, loss_fc)
        string1 = ('Stock {} Epoch[{}/{}] | train(mse):{:.6f}, mape:{:.6f} | '
                   'test(mse):{:.6f}, mape:{:.6f}\n'
                   .format(stock_name, epoch + 1, epochs,
                           train_mse[epoch],train_mape[epoch],
                           val_mse[epoch], val_mape[epoch], filename))
        string2 = '{}\n'.format(filename)
        print(string1)
        if epoch == epochs - 1:
            to_output.write(string1)
            to_output.write(string2)
    to_output.close()
    plt.plot(train_mse, label='training_error')
    plt.plot(val_mse, label='validation_error')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.title("Error vs Epoch")
    plt.legend()
    plt.savefig(filename + stock_name + '_mse.png')
    plt.show()

    plt.plot(train_mape, label='training_error')
    plt.plot(val_mape, label='validation_error')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Percentage Error")
    plt.title("Error vs Epoch")
    plt.legend()
    plt.savefig(filename + '_mape.png')
    plt.show()
