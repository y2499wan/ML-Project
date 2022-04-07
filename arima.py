import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF

df = pd.read_csv("TSLA.csv")
X = df.iloc[:, 1:]
X_simp = df['Close']
Y = df["Close"]
x, y = X_simp.to_numpy(), Y.to_numpy()

# train-test split 50%-50%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
print(x_train.shape, y_train.shape, x_test.shape, sep='\n')


# -------------------------------------------------------------------------------------


def diff(timeseries):
    timeseries_diff1 = np.diff(timeseries, axis=0)
    timeseries_diff2 = np.diff(timeseries_diff1, axis=0)

    timeseries_adf = ADF(timeseries.tolist())
    timeseries_diff1_adf = ADF(timeseries_diff1.tolist())
    timeseries_diff2_adf = ADF(timeseries_diff2.tolist())

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()


diff(x_train)


def autocorrelation(timeseries, lags):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()


diff1 = np.diff(x_train, axis=0)
diff1 = np.nan_to_num(diff1)
diff2 = np.diff(diff1, axis=0)
diff2 = np.nan_to_num(diff2)
autocorrelation(diff1, 15)


def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit()


decomposition = seasonal_decompose(x_train, period=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = np.nan_to_num(trend)
seasonal = np.nan_to_num(seasonal)
residual = np.nan_to_num(residual)

# 趋势序列模型训练
trend_model = ARIMA_Model(trend, (2, 0, 3))
trend_fit_seq = trend_model.fittedvalues
trend_predict_seq = trend_model.predict()

# 残差序列模型训练
residual_model = ARIMA_Model(residual, (4, 0, 4))
residual_fit_seq = residual_model.fittedvalues
residual_predict_seq = residual_model.predict()

# 拟合训练集
fit_seq = pd.Series(seasonal)
fit_seq = fit_seq.add(trend_fit_seq, fill_value=0)
fit_seq = fit_seq.add(residual_fit_seq, fill_value=0)

'''
plt.plot(fit_seq, color='red', label='fit_seq')
plt.plot(x_train, color='blue', label='purchase_seq_train')
plt.legend(loc='best')
plt.show()
'''

fit_seq - x_train

# 预测测试集
# 这里测试数据的周期性是根据seasonal对象打印的结果，看到里面的数据每7天一个周期，2014-08-01~2014-08-10的数据正好和2014-04-04~2014-04-13的数据一致
decomposition = seasonal_decompose(x_test, period=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = np.nan_to_num(trend)
seasonal = np.nan_to_num(seasonal)
residual = np.nan_to_num(residual)

predict_seq = pd.Series(seasonal)
predict_seq = predict_seq.add(trend_predict_seq)
predict_seq = predict_seq.add(residual_predict_seq)

'''
plt.plot(predict_seq, color='red', label='predict_seq')
plt.plot(x_test, color='blue', label='purchase_seq_test')
plt.legend(loc='best')
plt.show()
'''

train_error = fit_seq / x_train - 1
val_error = predict_seq / x_test - 1
e1 = np.mean(train_error)
e2 = np.mean(val_error)
print('The average training error is {:.2f}%. \nThe average testing error is {:.2f}%.'.format(e1 * 100, e2 * 100))
