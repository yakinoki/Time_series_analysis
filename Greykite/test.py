# Greykiteライブラリのインストール
!pip install greykite

# 必要なモジュールのインポート
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.simple_silverkite.silverkite_estimator import SilverkiteEstimator
from greykite.framework.templates.simple_silverkite.silverkite_simple_gcv import SimpleGCV

# サンプルデータの生成
import pandas as pd
import numpy as np

np.random.seed(42)
date_rng = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
y = np.random.normal(loc=0, scale=1, size=len(date_rng))

df = pd.DataFrame({"timestamp": date_rng, "value": y})

# 設定の定義
forecast_horizon = 30  # 予測の期間
config = ForecastConfig(
    forecast_horizon=forecast_horizon,
    coverage=0.95,
    metadata_param="auto",
    agg_level="auto",
    model_template=ModelTemplateEnum.SILVERKITE.name,
    holiday_effect_prior_scale=10,
    yearly_seasonality_order=8,
    weekly_seasonality_order=3,
    regularization_alpha=0.1,
)

# Forecasterの初期化とデータのフィット
forecaster = Forecaster()
result = forecaster.run_forecast(df, config)

# 予測結果の取得
forecast = result.forecast

# 予測結果のプロット
result.plot()

# 予測結果の評価
mse = result.mean_squared_error()
print(f"Mean Squared Error: {mse}")
