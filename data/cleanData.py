#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:34:25 2022

@author: wangcatherine
"""

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.
        
    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton
    
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n, axis=0)
    
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    
    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    
    # inverse fourier transform
    c_data = np.fft.ifft(fft, axis=0)
    
    if to_real:
        c_data = c_data.real
    
    return c_data

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
    data = fft_denoiser(data,0.001,True)

    """https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_Transformer%2BTimeEmbedding.ipynb"""

    dpc = np.diff(data, axis=0) / data[:-1, :]
    c1 = np.diff(data, axis=0)
    """Percentage change"""

    #scaler = MinMaxScaler()
    #dpc1 = scaler.fit_transform(dpc)

    df_dpc = pd.DataFrame(dpc, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    output = filename + '_cleaned.csv'
    df_dpc.to_csv(output)

if __name__ == '__main__':
    clean_data('AAPL')