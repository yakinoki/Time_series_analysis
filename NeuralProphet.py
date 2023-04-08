import pandas as pd
from neuralprophet import NeuralProphet

# データの読み込み
df_train = pd.read_csv('dataset/raw_data.csv')
df_train.columns = ['ds', 'y']

model = NeuralProphet()
model.fit(df_train)
#df_testの最後の日付から何日間分未来の予測値を出すか。
future = model.make_future_dataframe(df_train, periods=60)
forecast = model.predict(future)
#予測の最終日から何日分出力するか。
result = forecast[['ds', 'yhat1']].tail(30)


# 予測結果を出力
print(result)
