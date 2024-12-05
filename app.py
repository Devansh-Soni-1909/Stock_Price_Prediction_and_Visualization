


# # 3rd 

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # A placeholder for credentials (for simplicity, you can use hardcoded ones)
# VALID_USERNAME = "user"
# VALID_PASSWORD = "pass"

# # Function to show stock prediction app after login
# def show_stock_prediction(username):
#     st.title('Stock Trend Prediction')

#     # Set the date range
#     start = '2018-12-31'
#     end = '2023-06-15'

#     # Stock ticker input from user
#     user_input = st.text_input('Enter Stock Ticker', 'AAPL', key='ticker_input')

#     try:
#         # Fetch data from Yahoo Finance using yfinance
#         df = yf.download(user_input, start=start, end=end)

#         # Display subheader and data description
#         st.subheader('Data from 2010 - 2019')
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

# # Function to show stock-related articles (placeholder)
# def stock_articles():
#     st.title("Stock Market News and Insights")

#     # Placeholder articles
#     st.subheader("1. How the stock market behaves during recessions")
#     st.write("Summary of how stock prices are impacted during different stages of a recession...")

#     st.subheader("2. Technology stocks to watch in 2024")
#     st.write("A list of tech stocks that could experience growth in 2024...")

#     st.subheader("3. Expert tips on stock diversification")
#     st.write("Insights from stock market experts on the importance of a diversified portfolio...")

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
import requests  # For real-time news API

# A placeholder for credentials (for simplicity, you can use hardcoded ones)
VALID_USERNAME = "user"
VALID_PASSWORD = "pass"

# Function to show stock prediction app after login
def show_stock_prediction(username):
    st.title('Stock Trend Prediction')

    # Set the date range
    start = '2018-12-31'
    end = '2023-06-15'

    # Stock ticker input from user
    user_input = st.text_input('Enter Stock Ticker', 'AAPL', key='ticker_input')

    try:
        # Fetch data from Yahoo Finance using yfinance
        df = yf.download(user_input, start=start, end=end)

        # Display subheader and data description
        st.subheader('Data from 2010 - 2019')
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
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Stock Prediction", "Stock Articles", "Logout"])

    if page == "Stock Prediction":
        show_stock_prediction(username)  # Run the stock prediction function
    elif page == "Stock Articles":
        stock_articles()  # Display stock-related articles
    elif page == "Logout":
        st.session_state['logged_in'] = False  # Log out
else:
    pass  # Run the login function


