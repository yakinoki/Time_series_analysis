# Optuna のinstallはpip

import statsmodels.api as sm
import matplotlib.pyplot as plt
import optuna as op
import itertools
import pandas as pd
import numpy as np


# データの読み込み
bill = pd.read_csv("sample.csv",dtype={'x02':'float64'})

