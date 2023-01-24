# Optuna のinstallはpip

import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna as op
import itertools
import pandas as pd
import numpy as np
from datetime import datetime


# 日付形式で読み込む
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('test_data.csv', index_col='ds', date_parser=dateparse, dtype='float')


# ローカルレベルモデルによる状態推定
model = sm.tsa.UnobservedComponents(data, 'local level')
result = model.fit()
fig = result.plot_components()
print(result.summary())
