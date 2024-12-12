




#  latest

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import requests  # For real-time news API
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# from sklearn.metrics import r2_score
# import numpy as np


# # A placeholder for credentials (for simplicity, you can use hardcoded ones)
# VALID_USERNAME = "user"
# VALID_PASSWORD = "pass"

# # Function to show stock prediction app after login
# def show_stock_prediction(username):
#     st.title('Stock Trend Prediction')

#     # Set the date range
#     start = '2018-05-12'
#     end = '2024-04-12'

#     # Stock ticker input from user
#     user_input = st.text_input('Enter Stock Ticker', 'AAPL', key='ticker_input')

#     try:
#         # Fetch data from Yahoo Finance using yfinance
#         df = yf.download(user_input, start=start, end=end)

#         # Display subheader and data description
#         st.subheader('Data from 2018 - 2024')
#         st.write(df.describe())

#         # Visualizations
#         st.subheader('Closing Price vs Time Chart')
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(df.Close)
#         st.pyplot(fig)

#         st.subheader('Closing Price vs Time Chart with 100MA')
#         ma100 = df.Close.rolling(100).mean()
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(ma100, label='100MA')
#         plt.plot(df.Close, label='Closing Price')
#         plt.legend()
#         st.pyplot(fig)

#         st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
#         ma200 = df.Close.rolling(200).mean()
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(ma100, label='100MA')
#         plt.plot(ma200, label='200MA')
#         plt.plot(df.Close, label='Closing Price')
#         plt.legend()
#         st.pyplot(fig)

#         # Splitting data into training and testing
#         data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#         data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

#         # MinMaxScaler
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data_training_array = scaler.fit_transform(data_training)

#         # Load model
#         model = load_model('keras_model.h5')

#         # Testing part
#         past_100_days = data_training.tail(100)
#         final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#         input_data = scaler.fit_transform(final_df)

#         # Preparing test data
#         x_test = []
#         y_test = []

#         for i in range(100, input_data.shape[0]):
#             x_test.append(input_data[i-100:i])
#             y_test.append(input_data[i, 0])

#         x_test, y_test = np.array(x_test), np.array(y_test)

#         # Predictions
#         y_predicted = model.predict(x_test)

#         # Rescaling to original values
#         scale_factor = 1 / scaler.scale_[0]
#         y_predicted = y_predicted * scale_factor
#         y_test = y_test * scale_factor

#         # Calculate MSE, RMSE, and MAPE ,R-squared value
#         mse = mean_squared_error(y_test, y_predicted)
#         rmse = np.sqrt(mse)
#         mape = mean_absolute_percentage_error(y_test, y_predicted) * 100
#         r2 = r2_score(y_test, y_predicted)

#         # Display the results in Streamlit
#         st.subheader('Model Performance Metrics')
#         st.write(f"Mean Squared Error (MSE): {mse}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse}")
#         st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")
#         st.write(f"R-squared (R²): {r2}")

#         # Final graph
#         st.subheader('Predictions vs Original')
#         fig2 = plt.figure(figsize=(12, 6))
#         plt.plot(y_test, 'b', label='Original Price')
#         plt.plot(y_predicted, 'r', label='Predicted Price')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.legend()
#         st.pyplot(fig2)

#     except Exception as e:
#         st.error(f"Error fetching data for ticker {user_input}: {e}")

# # Function to fetch and display real-time stock news
# def fetch_stock_news():
#     st.title("Real-Time Stock News")
    
#     # Fetch news using an API (e.g., NewsAPI.org)
#     api_key = "9bfa7f6c6dda446eae2434fb6fa5c4f0"
#     url = f"https://newsapi.org/v2/everything?q=stocks&apiKey={api_key}"
    
#     try:
#         response = requests.get(url)
#         news_data = response.json()
#         articles = news_data['articles']

#         # Display news articles
#         for article in articles[:5]:
#             st.subheader(article['title'])
#             st.write(article['description'])
#             st.write(f"[Read more]({article['url']})")
    
#     except Exception as e:
#         st.error(f"Error fetching stock news: {e}")

# # Function to show stock-related articles and a list of stock tickers
# def stock_articles():
#     st.title("Stock Market News and Insights")

#     st.subheader("Latest Stock Market News")
#     fetch_stock_news()  # Fetch real-time news

#     st.subheader("List of Company Stock Tickers")
    
#     # Fetch a list of major stock tickers using yfinance
#     tickers = yf.Tickers('AAPL MSFT TSLA GOOGL AMZN')
#     ticker_list = tickers.symbols
#     st.write(ticker_list)

# # Login Page Styling
# st.markdown("""
#     <style>
#     .background {
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         z-index: -1;
#     }
#     .login-form {
#         position: absolute;
#         top: 20%;
#         left: 50%;
#         transform: translateX(-50%);
#         z-index: 1;
#         padding: 20px;
#         background-color: rgba(0, 0, 0, 0.5);
#         color: white;
#         border-radius: 10px;
#         box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Display the background image
# st.image("download.jpg", use_column_width=True)

# # Create a login form
# with st.form(key="login_form"):
#     st.markdown('<div class="login-form">', unsafe_allow_html=True)
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     submit_button = st.form_submit_button(label="Login")
#     st.markdown('</div>', unsafe_allow_html=True)

# # Handling form submission
# if submit_button:
#     if username == VALID_USERNAME and password == VALID_PASSWORD:
#         st.session_state['logged_in'] = True
#         st.success(f"Welcome, {username}!")
#     else:
#         st.error("Invalid credentials. Please try again.")

# # Main content routing
# if 'logged_in' not in st.session_state:
#     st.session_state['logged_in'] = False

# # If logged in, display main content, otherwise show login page
# if st.session_state['logged_in']:
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Stock Prediction", "Stock Articles", "Logout"])

#     if page == "Stock Prediction":
#         show_stock_prediction(username)  # Run the stock prediction function
#     elif page == "Stock Articles":
#         stock_articles()  # Display stock-related articles
#     elif page == "Logout":
#         st.session_state['logged_in'] = False  # Log out
# else:
#     pass  # Run the login function



import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Placeholder for credentials (for simplicity, you can use hardcoded ones)
VALID_USERNAME = "user"
VALID_PASSWORD = "pass"

# Function to show stock prediction app after login
def show_stock_prediction(username):
    st.title('Stock Trend Prediction')

    # Set the date range
    start = '2018-05-12'
    end = '2024-04-12'

    # Stock ticker input from user
    user_input = st.text_input('Enter Stock Ticker', 'AAPL', key='ticker_input')

    # Number of future days to predict
    forecast_days = st.number_input('Enter the number of days to forecast', min_value=1, max_value=365, value=30)

    try:
        # Fetch data from Yahoo Finance using yfinance
        df = yf.download(user_input, start=start, end=end)

        # Display subheader and data description
        st.subheader('Data from 2018 - 2024')
        st.write(df.describe())

        # Visualizations
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close)
        st.pyplot(fig)

        st.subheader('Closing Price vs Time Chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100MA')
        plt.plot(df.Close, label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100MA')
        plt.plot(ma200, label='200MA')
        plt.plot(df.Close, label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Splitting data into training and testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        # MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load model
        model = load_model('keras_model.h5')

        # Testing part
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        # Preparing test data
        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predictions
        y_predicted = model.predict(x_test)

        # Rescaling to original values
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Calculate MSE, RMSE, and MAPE, R-squared value
        mse = mean_squared_error(y_test, y_predicted)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_predicted) * 100
        r2 = r2_score(y_test, y_predicted)

        # Display the results in Streamlit
        st.subheader('Model Performance Metrics')
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")
        st.write(f"R-squared (R²): {r2}")

        # Verify if forecasting is done correctly
        st.subheader('Forecasting Verification')
        if r2 > 0.9:
            st.success('Forecasting is done correctly with high accuracy (R² > 0.9).')
        elif 0.7 < r2 <= 0.9:
            st.warning('Forecasting is reasonable (0.7 < R² <= 0.9), but can be improved.')
        else:
            st.error('Forecasting may not be accurate (R² <= 0.7), further improvement is needed.')

        # Predict future values for the next 'x' days
        last_100_days = input_data[-100:]
        last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))  # Ensure it matches model input shape
        
        predicted_future = []
        for _ in range(forecast_days):
            future_pred = model.predict(last_100_days)
            predicted_future.append(future_pred[0, 0])
            # Update the last 100 days to include the new prediction
            future_pred_reshaped = np.reshape(future_pred, (1, 1, 1))  # Ensure future_pred is reshaped correctly
            last_100_days = np.append(last_100_days[:, 1:, :], future_pred_reshaped, axis=1)
        
        predicted_future = np.array(predicted_future)
        predicted_future = predicted_future * scale_factor
        predicted_future = predicted_future * scale_factor

        # Display future predictions
        st.subheader(f'Forecast for the Next {forecast_days} Days')
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(np.arange(1, forecast_days+1), predicted_future, label='Forecasted Price')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig3)

        # Final graph
        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error fetching data for ticker {user_input}: {e}")

# Function to fetch and display real-time stock news
def fetch_stock_news():
    st.title("Real-Time Stock News")
    
    # Fetch news using an API (e.g., NewsAPI.org)
    api_key = "9bfa7f6c6dda446eae2434fb6fa5c4f0"
    url = f"https://newsapi.org/v2/everything?q=stocks&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        news_data = response.json()
        articles = news_data['articles']

        # Display news articles
        for article in articles[:5]:
            st.subheader(article['title'])
            st.write(article['description'])
            st.write(f"[Read more]({article['url']})")
    
    except Exception as e:
        st.error(f"Error fetching stock news: {e}")

# Function to show stock-related articles and a list of stock tickers
def stock_articles():
    st.title("Stock Market News and Insights")

    st.subheader("Latest Stock Market News")
    fetch_stock_news()  # Fetch real-time news

    st.subheader("List of Company Stock Tickers")
    
    # Fetch a list of major stock tickers using yfinance
    tickers = yf.Tickers('AAPL MSFT TSLA GOOGL AMZN')
    ticker_list = tickers.symbols
    st.write(ticker_list)

# Login Page Styling
st.markdown("""
    <style>
    .background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }
    .login-form {
        position: absolute;
        top: 20%;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Display the background image
st.image("download.jpg", use_column_width=True)

# Create a login form
with st.form(key="login_form"):
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit_button = st.form_submit_button(label="Login")
    st.markdown('</div>', unsafe_allow_html=True)

# Handling form submission
if submit_button:
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        st.session_state['logged_in'] = True
        st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid credentials. Please try again.")

# Main content routing
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# If logged in, display main content, otherwise show login page
if st.session_state['logged_in']:
    selected_page = st.sidebar.selectbox("Select Page", ("Stock Prediction", "Stock Market News", "Logout"))

    if selected_page == "Stock Prediction":
        show_stock_prediction(username)
    elif selected_page == "Stock Market News":
        stock_articles()
    elif selected_page == "Logout":
        st.session_state['logged_in'] = False
        st.success("You have logged out successfully.")
else:
    st.title("Please log in to access the stock prediction and news features.")




