import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://media.istockphoto.com/id/1399521171/video/analyzing-digital-data-blue-version-loopable-statistics-financial-chart-economy.jpg?s=640x640&k=20&c=jwtU00hvkCK1UShIzRDrmIwJpft4LkOyFagEqMKvidM=");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

#Sidebar
st.markdown("""
  <style>
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    },
    .st-emotion-cache-uf99v8 {
        align-items: start
        padding: 2rem 1.5rem;
    }
    .text-input {
        background: transparent;
        border: 0;
        font-size: 20px;
        color: #f2f2f2;
        height: 30px;
        line-height: 30px;
        outline: none !important;
        width: 100%;
        }
    .st-emotion-cache-gh2jqd{
        padding: 0px 0px 0px; 
    }
    .e1f1d6gn4{
        margin: 0px;
    }
  </style>
""", unsafe_allow_html=True)
st.markdown(background_image, unsafe_allow_html=True)


with st.sidebar:
    st.header("About App")
    st.subheader(":red[Presented By:]")
    st.write("* PATEL PREET")
    st.write("* TANDEL MANASVI")
    st.write("* PATEL SUJAL")
    st.write("* DALAL KUSH")
    st.markdown("---")  # Separator line

    # Section 2: Data Exploration Controls
    st.subheader(":red[Guided by]")
    st.write("Abdul Aziz Md, Master Trainer, Edunet Foundation.")

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title(":red[Stream Stock Trend]")

user_input = st.text_input('Enter stock ticker:','AAPL') 

# Define a function to load the dataset
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#Discribing Data
data = load_data(user_input)
df=data
# st.write(df.head())

#Visulation Data
st.subheader(":red[Close price vs Time chart]")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.title("India Stock Price")
plt.xlabel("Days")
plt.ylabel("Price (INR)")
plt.grid(True)
st.pyplot(fig)


#100EMA Visulation
st.subheader(":red[Close price vs Time chart with 100EMA]")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
plt.title("India Stock Price")
plt.xlabel("Days")
plt.ylabel("Price (INR)")
plt.grid(True)
st.pyplot(fig)

#100EMA Visulation
st.subheader(":red[Close price vs Time chart with 100EMA & 200EMA]")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma200,'r')
plt.plot(ma100,'y')
plt.title("India Stock Price")
plt.xlabel("Days")
plt.ylabel("Price (INR)")
plt.grid(True)
st.pyplot(fig)

#Spliting data into traing and testing
train = pd.DataFrame(data[0:int(len(df)*0.70)])
test = pd.DataFrame(data[int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values
data_training_array = scaler.fit_transform(train_close)

#Spliting data into X-train and y-train
x_train = []
y_train = [] 
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train) 

#Load my model
model = load_model('keras_model.h5')

#Testing part
past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
   x_test.append(input_data[i-100: i])
   y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scale_factor = 1/0.00041967
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

#Final Pridiction Graph
st.subheader(":red[Predictions VS Original]")
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig2)