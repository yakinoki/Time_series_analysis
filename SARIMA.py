# Optuna のinstallはpip

import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna as op
import itertools
import pandas as pd
import numpy as np


# データの読み込み
bill = pd.read_csv("sample.csv",dtype={'x02':'float64'})

# データの整理
index = pd. date_range("2011-03-31","2015-03-31",freq="M")
bill.index = index
del bill["Month"]


# モデルの構築
def selectparameter(DATA,s):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
    parameters = []
    BICs = np.array([])
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(DATA,
                                            order=param,
                                            seasonal_order=param_seasonal)
                results = mod.fit()
                parameters.append([param, param_seasonal, results.bic])
                BICs = np.append(BICs,results.bic)
            except:
                continue
    return parameters[np.argmin(BICs)]


# パラメータ決定
selectparameter(bill, 12)

#モデルの当てはめ
SARIMA_bill = sm.tsa.statespace.SARIMAX(bill,order=(0, 1, 1),seasonal_order=(1, 1, 0, 12)).fit()

#predに予測データを代入する
pred = SARIMA_bill.predict("2015-04-30", "2018-01-31")

#predデータともとの時系列データの可視化
plt.plot(bill)
plt.plot(pred)
plt.show()
    