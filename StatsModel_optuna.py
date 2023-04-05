import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna
import pandas as pd

# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# 日付形式で読み込む
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('dataset/raw_data.csv', index_col='ds', date_parser=dateparse, dtype='float')

def objective(trial):
    # パラメータの範囲を定義
    seasonal = trial.suggest_int('seasonal', 2, 24)
    smoothing_level = trial.suggest_uniform('smoothing_level', 0, 1)
    
    # ローカルレベルモデルによる状態推定
    model = sm.tsa.UnobservedComponents(data, 'local level', seasonal=seasonal)
    result = model.fit(smoothing_level=smoothing_level)
    
    # 目的関数としてMAPEを使用
    pred = result.predict("2017-12-01", "2017-12-31")
    actual = data["2017-12-01":"2017-12-31"]
    #mape = abs((actual - pred) / actual).mean()
    mape = pred.mean()
    
    return mape

# optunaによるパラメータチューニング
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 最適解の取得
best_params = study.best_params
print(f"best params: {best_params}")

# 最適パラメータでのモデル構築と予測
model = sm.tsa.UnobservedComponents(data, 'local level', seasonal=best_params["seasonal"])
result = model.fit(smoothing_level=best_params["smoothing_level"])
pred = result.predict("2018-01-01", "2018-08-31")

# 実データと予測結果の図示
plt.plot(data)
plt.plot(pred, "r")
plt.show()
