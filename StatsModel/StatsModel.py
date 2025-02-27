import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# 日付形式で読み込む
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('../dataset/test_data.csv', index_col='ds', date_parser=dateparse, dtype='float')

# 祝日データの読み込み（仮定）
holidays = pd.read_csv('../dataset/holidays.csv', index_col='date', date_parser=dateparse, dtype='int')

# データフレームに祝日情報の列を追加
data['holiday'] = holidays['is_holiday']

# ローカルレベルモデルによる状態推定
model = sm.tsa.UnobservedComponents(data['y'], 'local level', seasonal=12)
result = model.fit()
#fig = result.plot_components()
print(result.summary())
#fig.savefig("img.png")

# 予測
pred = result.predict("2018-12-01", "2018-12-31")
#actual = data["2017-12-01":"2017-12-31"]

# 実データと予測結果の図示
plt.plot(data)
plt.plot(pred, "r")
plt.show()
