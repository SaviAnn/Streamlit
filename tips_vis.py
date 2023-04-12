import yfinance as yf
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
st.write("""
# Чаевые в ресторане
### Ниже представлены визуализация датасета tips с помощью библиотеки seaborn
""")
tips=pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
fig = plt.figure(figsize=(10, 4))
sns.histplot(x="total_bill", data=tips)

st.pyplot(fig)
fig1 = plt.figure(figsize=(10, 4))
sns.scatterplot(x='tip', 
                y='total_bill',
                data=tips)
st.pyplot(fig1)
fig2 = plt.figure(figsize=(10, 4))
sns.scatterplot(x='tip', 
                y='total_bill', 
                hue='size',
                data=tips)
st.pyplot(fig2)
fig3 = plt.figure(figsize=(10, 4))
sns.barplot(x='day', 
                y='total_bill',
                data=tips)
st.pyplot(fig3)
fig4 = plt.figure(figsize=(10, 4))
sns.scatterplot(x='tip', 
                y='day', 
                hue='sex',
                data=tips)
st.pyplot(fig4)
fig5= plt.figure(figsize=(10, 4))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time")
st.pyplot(fig5)
#fig6 = plt.figure(figsize=(10, 4))
fig6, axes = plt.subplots(1, 2, figsize=(10, 6))
fig6.suptitle('Dinner and Lunch Tips')

sns.histplot(ax=axes[0], x='tip', data=tips[tips['time']=='Lunch'], kde=True, hue='time')

sns.histplot(ax=axes[1], x='tip', data=tips[tips['time']=='Dinner'],kde=True, hue='time')
st.pyplot(fig6)
#fig7 = plt.figure(figsize=(10, 4))
fig7, axes = plt.subplots(1, 2, figsize=(10, 6))
fig7.suptitle('Gender Comparison')

sns.scatterplot(ax=axes[0], x='tip', y='total_bill', data=tips[tips['sex']=='Female'], hue='smoker').set(title='Female')

sns.scatterplot(ax=axes[1], x='tip', y='total_bill', data=tips[tips['sex']=='Male'], hue='smoker').set(title='Male')
fig.suptitle('Gender Comparison')

sns.scatterplot(ax=axes[0], x='tip', y='total_bill', data=tips[tips['sex']=='Female'], hue='smoker').set(title='Female')

sns.scatterplot(ax=axes[1], x='tip', y='total_bill', data=tips[tips['sex']=='Male'], hue='smoker').set(title='Male')
st.pyplot(fig7)