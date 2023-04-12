import yfinance as yf
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
st.write("""
# Чаевые в ресторане :coffee: 
### Ниже представлена визуализация датасета **tips** с помощью библиотеки _seaborn_
""")
tips=pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
st.write(tips)
st.write(""" Как видно из диаграммы ниже, средний счет находится в диапазоне 20-30 долларов
""")
         #define seaborn background colors
sns.set_style('darkgrid')
sns.color_palette("rocket", as_cmap=True)

fig = plt.figure(figsize=(10, 4))
sns.histplot(x="total_bill", data=tips)

st.pyplot(fig)
st.write("""
### Связь между чаевыми и общим счетом
Заметна линейная прямопропорциональная зависимость
""")
fig1 = plt.figure(figsize=(10, 4))
sns.regplot(x="total_bill", y="tip", data=tips);
st.pyplot(fig1)
fig2 = plt.figure(figsize=(10, 4))
sns.scatterplot(x='tip', 
                y='total_bill', 
                hue='size',
                data=tips)
st.write("""
### Дополнительно выделим цветом количество блюд в заказе
""")
st.pyplot(fig2)
st.write("""
### Зависимость размера счета от дня недели
В целом, отклонения от среднего чека незначительные, но видно, что счет в выходные больше :arrow_heading_up: , чем в будние дни
""")
fig3 = plt.figure(figsize=(10, 4))
sns.barplot(x='day', 
                y='total_bill',
                data=tips)
st.pyplot(fig3)
st.write("""
### Зависимость чаевых от дня недели
В выходные люди более щедрые, но зависимость от гендера увидеть сложно
""")
fig4 = plt.figure(figsize=(10, 4))
sns.scatterplot(x='tip', 
                y='day', 
                hue='sex',
                data=tips)
st.pyplot(fig4)
st.write("""
### Сумма всех счетов за каждый день с разбивков по приемам пищи
Можно заметить, что в выходные обедов нет
""")
fig5= plt.figure(figsize=(10, 4))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time")
st.pyplot(fig5)

st.write("""
#### Сравнение распределения графиков распрелеления чаевых, в зависимости от приема пищи lunch/dinner 
""")
fig6, axes = plt.subplots(1, 2, figsize=(10, 6))
fig6.suptitle('Lunch and Dinner Tips')

sns.histplot(ax=axes[0], x='tip', data=tips[tips['time']=='Lunch'], kde=True, hue='time')

sns.histplot(ax=axes[1], x='tip', data=tips[tips['time']=='Dinner'],kde=True, hue='time')
st.pyplot(fig6)
st.write("""
#### На последнем графике можно увидеть связь между счетом и чаевыми с разбивкой по полу и привычке курить :smoking:
""")
fig7, axes = plt.subplots(1, 2, figsize=(10, 6))

fig7.suptitle('Сравнение по чаевым между женщинами и мужчинами')

sns.scatterplot(ax=axes[0], x='tip', y='total_bill', data=tips[tips['sex']=='Female'], hue='smoker').set(title='Female')

sns.scatterplot(ax=axes[1], x='tip', y='total_bill', data=tips[tips['sex']=='Male'], hue='smoker').set(title='Male')
st.pyplot(fig7)
