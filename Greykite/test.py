import yaml
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.model_templates import ModelTemplateEnum

# YAMLファイルの読み込み
with open("config/greykite_config.yml", "r") as file:
    config_dict = yaml.safe_load(file)

# ForecastConfigの構築
config = ForecastConfig(
    forecast_horizon=config_dict["forecast_horizon"],
    coverage=config_dict["coverage"],
    metadata_param=config_dict["metadata_param"],
    agg_level=config_dict["agg_level"],
    model_template=getattr(ModelTemplateEnum, config_dict["model_template"]),
    holiday_effect_prior_scale=config_dict["holiday_effect_prior_scale"],
    yearly_seasonality_order=config_dict["yearly_seasonality_order"],
    weekly_seasonality_order=config_dict["weekly_seasonality_order"],
    regularization_alpha=config_dict["regularization_alpha"],
)

# Forecasterの初期化とデータのフィット
forecaster = Forecaster()
result = forecaster.run_forecast(df, config)

# 予測結果のプロット
result.plot()

# プロットを表示
import matplotlib.pyplot as plt
plt.show()
