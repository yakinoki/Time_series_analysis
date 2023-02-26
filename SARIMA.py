# Optuna のinstallはpip

import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna as op
import itertools
import pandas as pd
import numpy as np
import yaml

#with open('config/config_sarima_month.yml','r',encoding="utf-8") as yml:
with open('config/config_sarima_date.yml','r',encoding="utf-8") as yml:
    config = yaml.safe_load(yml) 
    csv = config["csv"]      
    freq = config['freq']
    freq_per = config['freq_per']
    train_start_date = config['train_start_date']
    train_end_date = config['train_end_date']
    pred_start_date = config['pred_start_date']
    pred_end_date = config['pred_end_date']


# データの読み込み
bill = pd.read_csv(csv,dtype={'x02':'float64'})




# データの整理
index = pd.date_range(train_start_date,train_end_date,freq=freq)
bill.index = index
print(bill)

del bill[freq_per]


# モデルの構築
# p,d,qはそれぞれ「季節性自己相関」「季節性導出」「季節性移動平均」という。
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

#predに予測期間を代入する
pred = SARIMA_bill.predict(pred_start_date, pred_end_date)

#predデータともとの時系列データの可視化
plt.plot(bill)
plt.plot(pred)
plt.show()
