from typing import Any
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from TransformerNetwork import *
from t2v import *
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from hyperparameters import *
from test_methods import *

# =======================================================================
df = pd.read_csv(stock_name + "_cleaned.csv")
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
df = scaler_x.fit_transform(df)
x_raw = df[:, 1:]
y_raw = df[:, 4].reshape(-1, 1)
x_slide, y_slide = sliding_window(x_raw, y_raw, window_size)
x_train, x_test, y_train, y_test = train_test_split(x_slide, y_slide, test_size=0.2, random_state=1, shuffle=False)
x_train = torch.tensor(x_train.astype(np.float32))
y_train = torch.tensor(y_train.astype(np.float32))
x_val = torch.tensor(x_test.astype(np.float32))
y_val = torch.tensor(y_test.astype(np.float32))
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)


# initialize model
model = Transformer(dim_val, dim_attn, input_size, batch_size, enc_seq_len, dec_seq_len, output_sequence_length,
                    n_decoder_layers, n_encoder_layers, n_heads, time_embed_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fc = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
to_output = open('output.txt', 'a')
run_epoch(epochs, to_output, train_dataset, val_dataset, model, optimizer, loss_fc, mae)
