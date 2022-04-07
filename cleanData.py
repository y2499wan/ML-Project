# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def clean_data(filename):
    path = filename + '.csv'
    df = pd.read_csv(path, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # TO-DOs
    # fill outliers, none with median
    # transform into min-max

    # find missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    """Fourier Transform can help remove the noise by converting the time series data into the 
    frequency domain, and from there, we can filter out the noisy frequencies. 
    Then, we can apply the inverse Fourier transform to obtain the filtered time series."""

    df[(df.Volume == 0)]

    data = df.iloc[:, 1:].to_numpy()

    """https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_Transformer%2BTimeEmbedding.ipynb"""

    dpc = np.diff(data, axis=0) / data[:-1, :] * 100
    c1 = np.diff(data, axis=0)
    """Percentage change"""

    scaler = MinMaxScaler()
    dpc1 = scaler.fit_transform(dpc)

    df_dpc = pd.DataFrame(dpc1, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    output = filename + '_cleaned.csv'
    df_dpc.to_csv(output)

if __name__ == '__main__':
    clean_data('AMZN')