# Optuna のinstallはpip

import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna as op
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# 日付形式で読み込む
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('test_data.csv', index_col='ds', date_parser=dateparse, dtype='float')


# ローカルレベルモデルによる状態推定
model = sm.tsa.UnobservedComponents(data, 'local level', seasonal=12)
result = model.fit()
#fig = result.plot_components()
print(result.summary())
#fig.savefig("img.png")

# 予測
pred = result.predict("2019-01-01", "2019-08-31")

# 実データと予測結果の図示
plt.plot(data)
plt.plot(pred, "r")
plt.show()


