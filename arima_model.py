import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def train_arima(train_data, test_data, scaler, order=(5,1,0)):
    """
    Train ARIMA model and make predictions.
    
    Parameters:
    train_data (DataFrame): The training data
    test_data (DataFrame): The test data
    scaler (MinMaxScaler): Scaler to inverse the predictions
    order (tuple): The order of ARIMA (p,d,q)
    
    Returns:
    arima_predictions (array): Predicted stock prices
    arima_rmse (float): Root Mean Squared Error of the model
    """
    
    # Fit ARIMA model
    model_arima = ARIMA(train_data, order=order)
    model_arima_fit = model_arima.fit()

    # Make predictions
    arima_predictions = model_arima_fit.forecast(steps=len(test_data))
    arima_predictions = scaler.inverse_transform(arima_predictions.reshape(-1, 1))

    # Calculate RMSE
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_predictions))

    return arima_predictions, arima_rmse
