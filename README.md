# Time_series_analysis

You can use the following command to install the necessary packages.

```sh
pip install -r requirements.txt
```

## SARIMA
Test code for SARIMA model

The SARIMA (Seasonal Autoregressive Integrated Moving Average) model is a statistical model used for forecasting time series data, and it can decompose observed values into components such as trend, seasonality, and white noise (residuals). Here's a general logic for how the SARIMA model separates observed values:

1. Differencing:
The first step in the SARIMA model is differencing the data. This is done to reduce the trend component. The order of differencing is adjusted to ensure data stationarity. If the trend is already a stationary process, differencing may not be necessary.

2. Seasonal Identification:
If there is seasonality in the data, the SARIMA model needs to capture it. Identify the seasonal pattern (e.g., yearly, quarterly, monthly), and set the seasonal orders (P, D, Q) accordingly.

3. Model Specification:
The SARIMA model is defined by combining elements of AR (AutoRegressive), I (Integrated), MA (Moving Average), and seasonal components. Choose the appropriate orders for AR (p), I (d), MA (q), and seasonal AR (P), seasonal I (D), seasonal MA (Q). This is typically done using plots of autocorrelation and partial autocorrelation functions of the data.

4. Model Estimation:
Estimate the parameters of the selected model. This is done using statistical methods such as maximum likelihood estimation.

5. Model Diagnosis:
Examine the residuals of the estimated model and assess the goodness of fit. Residuals should ideally behave like white noise. Model diagnosis involves plotting the autocorrelation and partial autocorrelation functions of the residuals, checking for normality of the residuals, and other diagnostic checks.

6. Forecasting:
Finally, use the SARIMA model to predict future values. Utilize the model parameters to make forecasts that account for trends, seasonality, and residuals.



## StatsModel
Test code for state-space model

In a state-space model, there are two variables: the state and the observed value. The observed value at time $t$, denoted as $y_t$, is defined to be generated from the state $x_t$. Additionally, the state $x_t$ is determined solely based on the previous state, $x_{t-1}$. These relationships can be expressed through equations as follows:
$$x_t = g(x_{t-1},w_t)$$ 
$$y_t = f(x_{t-1},v_t)$$
The equation concerning the state variable $x$ is referred to as the state equation, while the equation concerning the observed value $y$ is referred to as the observation equation.

## Prophet
Test code for Prophet model
In Prophet, time series data is conceptualized as having the following components:

$g(t)$: Trend function

$s(t)$: Seasonal variation

$h(t)$: Holiday effect

$ε_t$: Error term

Furthermore, time series data is modeled as the sum of these components, and it is constructed with the following formula:

$$y(t) = g(t) + s(t) + h(t) + ε_t.$$






## NeuralProphet.py
Test code for NeuralProphet model.
NeuralProphet is a library released by Facebook, which is a time series data forecasting model based on the combination of Prophet and AR-Net, a self-regressive neural network model.
