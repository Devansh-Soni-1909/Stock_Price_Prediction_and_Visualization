import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def train_lr(X_train, y_train, X_test, y_test, scaler):
    """
    Train Linear Regression model and make predictions.
    
    Parameters:
    X_train (array): Training data features
    y_train (array): Training data labels
    X_test (array): Test data features
    y_test (array): Test data labels
    scaler (MinMaxScaler): Scaler to inverse the predictions
    
    Returns:
    lr_predictions (array): Predicted stock prices
    lr_rmse (float): Root Mean Squared Error of the model
    """
    
    # Create and train Linear Regression model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # Make predictions
    lr_predictions = model_lr.predict(X_test)
    lr_predictions = scaler.inverse_transform(lr_predictions.reshape(-1, 1))

    # Calculate RMSE
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

    return lr_predictions, lr_rmse
