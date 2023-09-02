# Optuna のinstallはpip

import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna as op
import itertools
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error

with open('config/config_sarima_month.yml','r',encoding="utf-8") as yml:
#with open('config/config_sarima_date.yml','r',encoding="utf-8") as yml:
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

# インデックスデータの作成
index = pd.date_range(train_start_date,train_end_date,freq=freq)
# インデックスデータの代入
bill.index = index
print(bill)

# freq_perカラムの削除
del bill[freq_per]

# 最適なパラメータを求める
def objective(trial):
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 2)
    seasonal_p = trial.suggest_int('seasonal_p', 0, 2)
    seasonal_d = trial.suggest_int('seasonal_d', 0, 2)
    seasonal_q = trial.suggest_int('seasonal_q', 0, 2)
    s = 12 # 季節性の周期

    try:
        mod = sm.tsa.statespace.SARIMAX(bill,
                                        order=(p, d, q),
                                        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, s))
        results = mod.fit()
        pred = results.predict(pred_start_date, pred_end_date)
        # 実際の値
        actual = bill.loc[train_start_date:train_end_date]
        # 評価用の値
        evaluate = pred.loc[pred_start_date:train_end_date]
        # 平均絶対誤差の計算
        mae = mean_absolute_error(actual, evaluate)
    except:
        mae = float('inf')
    return mae

study = op.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


# 最適なパラメータを用いてモデルを当てはめる
params = study.best_params
SARIMA_bill = sm.tsa.statespace.SARIMAX(bill, order=(params['p'], params['d'], params['q']), 
                                        seasonal_order=(params['seasonal_p'], params['seasonal_d'], params['seasonal_q'], 12)).fit()

#予測
pred = SARIMA_bill.predict(pred_start_date, pred_end_date)

#predデータともとの時系列データの可視化
plt.plot(bill)
plt.plot(pred)
plt.show()
