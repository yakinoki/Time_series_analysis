import pandas as pd
import warnings
from prophet import Prophet
import jpholiday

warnings.filterwarnings("ignore")

# データの読み込み
df_train = pd.read_csv('dataset/raw_data.csv')
df_train.columns = ['ds', 'y']

# 祝日効果を追加する列を作成
df_train['holiday'] = pd.to_datetime(df_train['ds']).apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)

# データを訓練用とテスト用に分割する関数
def data_split(df, test_size):
    n = len(df)
    train_df = df.iloc[:int(n*(1-test_size))]
    test_df = df.iloc[int(n*(1-test_size)):]
    return train_df, test_df

# データを訓練用とテスト用に分割
train_df, test_df = data_split(df_train, test_size=0.2)

model = Prophet()

# 祝日効果を追加
model.add_regressor('holiday')

model.fit(train_df)

# df_testの最後の日付から何日間分未来の予測値を出すか。
future = model.make_future_dataframe(periods=60)

# 祝日効果の列を作成
future['holiday'] = future['ds'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)

forecast = model.predict(future)

# 予測の最終日から何日分出力するか。
def get_forecast(forecast, num_days):
    result = forecast[['ds', 'yhat']].tail(num_days)
    return result

# テストデータに対する予測を行う
test_forecast = model.predict(test_df)

# 予測結果を出力
result = get_forecast(test_forecast, 30)
print(result)
