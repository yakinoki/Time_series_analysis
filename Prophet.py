import pandas as pd
import warnings
from prophet import Prophet
import holidays

warnings.filterwarnings("ignore")

# データの読み込み
df_train = pd.read_csv('dataset/raw_data.csv')
df_train.columns = ['ds', 'y']

# 祝日の取得
jp_holidays = holidays.Japan(years=range(1970, 2024))  # 祝日の範囲は必要に応じて調整してください

# 祝日効果を追加するためのカスタム関数を定義
def add_holiday_effect(ds):
    date = pd.to_datetime(ds)
    if date in jp_holidays:
        return 1
    else:
        return 0

# 祝日効果を追加する列を作成
df_train['holiday'] = df_train['ds'].apply(add_holiday_effect)

model = Prophet()

# 祝日効果を追加
model.add_regressor('holiday')

model.fit(df_train)

# df_testの最後の日付から何日間分未来の予測値を出すか。
future = model.make_future_dataframe(periods=60)

# 祝日効果の列を作成
future['holiday'] = future['ds'].apply(add_holiday_effect)

forecast = model.predict(future)

# 予測の最終日から何日分出力するか。
result = forecast[['ds', 'yhat']].tail(30)

# 予測結果を出力
print(result)
