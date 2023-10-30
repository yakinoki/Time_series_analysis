import numpy as np
from ripser import Rips
import matplotlib.pyplot as plt

# 時系列データを生成
np.random.seed(0)
time_series_data = np.random.random(100)

# グラフ化
plt.figure(figsize=(10, 4))
plt.plot(time_series_data, label="Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Time Series Data")
plt.legend()
plt.show()

# 1次元のデータを2次元に変換
time_series_data_2d = time_series_data[:, np.newaxis]

# パーシステントホモロジーを計算
rips = Rips()
dgms = rips.fit_transform(time_series_data_2d)

# パーシステントダイアグラムをプロット
rips.plot(dgms)
plt.show()
