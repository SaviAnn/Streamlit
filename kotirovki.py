import yfinance as yf
import streamlit as st
import pandas as pd
from PIL import Image

st.write("""
# Данные о котировках компании Apple
Ниже представлены графики цен при открытии торгов,закрытии торгов, а также объему 
""")
image = Image.open('stonks.jpeg')

st.image(image, caption='Current mood')

tickerSymbol = 'AAPL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2013-4-11', end='2023-4-11')
st.line_chart(tickerDf.Open)
st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)
