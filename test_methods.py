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


def run_epoch(epochs, train_dataset, val_dataset, model, optimizer, loss_fc, mae, to_output):
    train_mse_error = []
    val_mse_error = []  # float to double -> .astype(np.float32)
    train_mape_error = []
    val_mape_error = []  # float to double -> .astype(np.float32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    random_filename = 'results/' + str(uuid.uuid1())
    for epoch in range(epochs):
        train_mse, train_mape = epoch_train(train_loader, model, optimizer, loss_fc, mae)
        train_mse_error.append(train_mse)
        train_mape_error.append(train_mape)
        val_mse, val_mape = epoch_test(val_loader, model, loss_fc)
        val_mse_error.append(val_mse)
        val_mape_error.append(val_mape)
        string = ('Stock {} Epoch[{}/{}] | train(mse):{:.6f}, mape:{:.6f} | '
                  'test(mse):{:.6f}, mape:{:.6f}\n{}\n\n'
                  .format(stock_name, epoch + 1, epochs, train_mse, train_mape, val_mse, val_mape, random_filename))
        print(string, end='')
        if epoch == epochs - 1:
            to_output.write(string)
    to_output.close()
    plt.plot(train_mse_error, label='training_error')
    plt.plot(val_mse_error, label='validation_error')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.title("Error vs Epoch")
    plt.legend()
    plt.savefig(random_filename+'_mse.png')
    plt.show()

    plt.plot(train_mape_error, label='training_error')
    plt.plot(val_mape_error, label='validation_error')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Percentage Error")
    plt.title("Error vs Epoch")
    plt.legend()
    plt.savefig(random_filename+'_mape.png')
    plt.show()

