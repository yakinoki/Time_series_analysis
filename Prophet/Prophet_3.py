import pandas as pd
import warnings
from prophet import Prophet

warnings.filterwarnings("ignore")

# データの読み込み
df_train = pd.read_csv('dataset/raw_data.csv')
df_train.columns = ['ds', 'y']

# イベントフラグの読み込み
df_events = pd.read_csv('dataset/events.csv')

# データにイベントフラグをマージ
df_train = pd.merge(df_train, df_events, on='ds', how='left')

# イベントフラグの欠損値を0で埋める
df_train['event_flag'] = df_train['event_flag'].fillna(0)

# パラメータの辞書を作成
params = {
    'growth': 'linear',  # 成長モデル
    # 'changepoints': ['2023-01-01', '2023-04-01'],  # トレンド変化点の指定
    'n_changepoints': 25,  # トレンド変化点の数
    'seasonality_mode': 'additive',  # 季節性モデル（加法モデル）
    'holidays': None,  # 祝日や特別なイベント
    'seasonality_prior_scale': 10.0,  # 季節性の強さ
    'holidays_prior_scale': 10.0,  # 祝日の影響
    'changepoint_prior_scale': 0.05,  # トレンド変化点の影響
    'daily_seasonality': True  # 曜日に対する季節性をモデル化する
}

# Prophetモデルのインスタンスを作成
model = Prophet(**params)

# イベントフラグを追加
model.add_regressor('event_flag')

model.fit(df_train)

# 予測用のデータフレームを作成
future = model.make_future_dataframe(periods=60)

# 予測用のデータにイベントフラグをマージ
future = pd.merge(future, df_events, on='ds', how='left')

# イベントフラグの欠損値を0で埋める
future['event_flag'] = future['event_flag'].fillna(0)

forecast = model.predict(future)

# 予測の最終日から何日分出力するか
result = forecast[['ds', 'yhat']].tail(30)

# 予測結果を出力
print(result)
