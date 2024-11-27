# Stock_Price_Prediction_and_Visualization
This project focuses on predicting stock price trends using a Long Short-Term Memory (LSTM) model, a type of recurrent neural network (RNN) well-suited for time-series forecasting. The project is built with Python and utilizes Streamlit for creating an interactive web application

# Stock Price Prediction and Visualization

## 📊 Project Overview

This project aims to predict the future stock prices using deep learning techniques, specifically **LSTM (Long Short-Term Memory)**. It also provides a visualization interface where users can see historical stock data, moving averages, and compare the predicted prices with the actual stock prices. 

The model is trained on historical stock data and makes future predictions based on trends, using **Keras** for model training and **Streamlit** for creating an interactive web app.

## 🚀 Key Features

- **Stock Data Visualization**: 
  - Displays the historical stock data from Yahoo Finance.
  - Shows moving averages (100-day and 200-day) for better trend analysis.
  - A comparison of actual vs. predicted prices.
  
- **Stock Price Prediction**: 
  - Uses an **LSTM model** for predicting stock prices.
  - Shows predictions alongside the actual closing prices.

- **Interactive Web App**:
  - Built with **Streamlit**, enabling users to enter a stock ticker and visualize the trends and predictions.

## 💡 Technologies Used

- **Python**: The core programming language for data manipulation, training, and deployment.
- **Keras/TensorFlow**: For building and training the LSTM model.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting the stock data and predictions.
- **yfinance**: For fetching historical stock data from Yahoo Finance.
- **Streamlit**: For creating a user-friendly web interface.

## 🌐 Live Demo

You can explore the **Stock Price Prediction and Visualization** web app live by visiting [Streamlit Cloud](https://share.streamlit.io/).  

## 📅 Date Range for Prediction

The model is trained on stock data from **January 1, 2010**, to **December 31, 2019**, providing predictions for the future.

## 🧑‍💻 How to Use the Web App

1. Clone this repository to your local machine.
   
   ```bash
   git clone https://github.com/Devansh-Soni-1909/Stock_Price_Prediction_and_Visualization.git

Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit app locally:


streamlit run app.py

Enter any stock ticker symbol (e.g., AAPL for Apple) to get the stock's historical data and predictions.


⚙️ How the Model Works
The stock prediction model is based on LSTM, a type of recurrent neural network (RNN) that excels in sequential data tasks, like time series prediction. The model is trained on historical stock data to predict future stock prices. The following steps are involved:

Data Collection:

Historical stock data is fetched using yfinance.
Data Preprocessing:

The data is normalized using MinMaxScaler to scale the values between 0 and 1.
Model Training:

The LSTM model is built with Keras and trained on the data.
Prediction:

The trained model makes future price predictions based on the past data.
Visualization:

The Streamlit app allows users to visualize the stock data, the moving averages, and predictions.
📝 Requirements
Ensure you have the following Python packages installed:

bash
Copy code
pip install numpy pandas matplotlib yfinance keras streamlit
🤝 Contributing
We welcome contributions! Feel free to fork this project, submit issues, and create pull requests to enhance the functionality of the project.

Fork the repo
Create a new branch (git checkout -b feature/your-feature)
Make your changes
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/your-feature)
Create a new Pull Request
💬 License
This project is licensed under the MIT License - see the LICENSE file for details.
