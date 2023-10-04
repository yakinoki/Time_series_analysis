# Time_series_analysis

## ./SARIMA
Test code for SARIMA model

## ./StatsModel
Test code for state-space model

In a state-space model, there are two variables: the state and the observed value. The observed value at time $t$, denoted as $y_t$, is defined to be generated from the state $x_t$. Additionally, the state $x_t$ is determined solely based on the previous state, $x_{t-1}$. These relationships can be expressed through equations as follows:
$$x_t = g(x_{t-1},w_t)$$ 
$$y_t = f(x_{t-1},v_t)$$
The equation concerning the state variable $x$ is referred to as the state equation, while the equation concerning the observed value $y$ is referred to as the observation equation.

## ./Prophet
Test code for Prophet model

## NeuralProphet.py
Test code for NeuralProphet model
