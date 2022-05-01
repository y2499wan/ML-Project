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
df = pd.read_csv("data/cleaned_data/" + stock_name + "_cleaned.csv")
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
df = scaler_x.fit_transform(df)
x_raw = df[:, 1:]
y_raw = df[:, 4].reshape(-1, 1)
x_slide, y_slide = sliding_window(x_raw, y_raw, window_size)
train_test = train_test_split(x_slide, y_slide, test_size=0.2, random_state=1, shuffle=False)
x_train, x_test, y_train, y_test = [torch.tensor(_.astype(np.float32)) for _ in train_test]
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_test, y_test)

# initialize model
model = Transformer(dim_val, dim_attn, input_size, batch_size, enc_seq_len,
                    n_encoder_layers, n_heads, time_embed_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fc = torch.nn.MSELoss()
to_output = open('results/output.txt', 'a')
run_epoch(epochs, train_dataset, val_dataset, model, optimizer, loss_fc, to_output)
