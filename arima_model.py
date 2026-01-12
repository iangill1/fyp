import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def arima_model_forecast(price_data):
    data = price_data

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    #create copy of data frame to avoid modifying original data
    data = data.copy()
    #fit ARIMA model
    model = ARIMA(data['Close'], order=(1, 0, 0))
    arima_prediction = model.fit()
    #get predictions
    rmse = np.sqrt(mean_squared_error(data['Close'], arima_prediction.fittedvalues))

    return rmse