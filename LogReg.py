import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from PIL import Image
from scipy.stats import poisson
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from io import StringIO
st.write("""
 # Logistic Regression Visualization
 
 """)
image = Image.open('image.jpeg')

st.image(image, caption='Let`s try!')
# import plotly.figure_factory as ff
class LogReg:
   def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.rand(n_inputs)
        self.intercept_ = np.random.rand()

   def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.rand(n_inputs)
        self.intercept_ = np.random.rand()

        
   def fit(self, X, y):
     steps=500
     for i in range(steps):
          yhat = self.intercept_ + np.dot(X, self.coef_)
          sigmoid_yhat = sigmoid(yhat)
          error = sigmoid_yhat - y
          Lw_grad = np.dot(X.T, error) / len(y)
          Lw0_grad = np.mean(error)
          self.coef_ = self.coef_ - self.learning_rate * Lw_grad
          self.intercept_ = self.intercept_ - self.learning_rate * Lw0_grad
     return self.coef_, self.intercept_

   def predict(self, X):
         Y = np.dot(X, self.coef_) + self.intercept_
         return Y
#    def score(self, X, y):
#         return r2_score(y, X@self.coef_ + self.intercept_)

def sigmoid(t):
    return (1/(1+np.exp(-t)))



# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
   
#     # Can be used wherever a "file-like" object is accepted:
#     train = pd.read_csv(uploaded_file)
train = pd.read_csv('LogRegtrain.csv').drop('Unnamed: 0', axis=1)
test = pd.read_csv('LogRegtest.csv').drop('Unnamed: 0', axis=1)
st.write("""
### Have a look for the given dataset:

""")
st.write(train)
st.write("""
### It's necessary to normalize all columns except Y:

""")

scaler = RobustScaler()
train[['x1', 'x2', 'x3']] = scaler.fit_transform(train[['x1', 'x2', 'x3']])
test[['x1', 'x2', 'x3']] = scaler.fit_transform(test[['x1', 'x2', 'x3']])
df_aux = train.describe(include='all')
st.write(df_aux)
# Создаем экземпляр класса LinReg с параметрами
# скорости обучения 0.01 и количеством признаков 3
logreg = LogReg(learning_rate=0.2, n_inputs=3)

# Выбираем столбцы с признаками из датасета train
X_train = train[['x1', 'x2', 'x3']].values

# Выбираем столбец с целевой переменной из датасета train
y_train = train['y'].values

# Обучаем модель на датасете train
logreg.fit(X_train, y_train)
st.write("""
### Let's compare weights results:

""")
logreg.fit(X_train, y_train)
c=logreg.coef_
d=logreg.intercept_
index_values = ['row1', 'row2'] # свой индекс

# создаем датафрейм из данных, используя свой индекс
# df = pd.DataFrame({'w1': c[0], 'w2': c[1], 'w3': c[2], 'intercept': d}, index=index_values)
# df = df.rename_axis('features', axis=1)
df = pd.DataFrame.from_dict({'w1': c[0], 'w2': c[1], 'w3': c[2], 'intercept': d}, orient='index')
df.index.name = 'features'

df.rename(columns = {0:' Our own derivation'}, inplace = True )
# выводим таблицу


# Создаем матрицу признаков для предсказания
X_test = test[['x1', 'x2', 'x3']].values
lr = LogisticRegression()
lr.fit(X_train, y_train)
a=lr.coef_
b=lr.intercept_
df["From Sklearn"] = [a[0,0], a[0,1], a[0,2], b[0]]
#    выводим таблицу
st.table(df)
# Получаем предсказания на матрице признаков X_test
y_pred = logreg.predict(X_test)
# выбираем колонки для визуализации
# columns = list(train.columns)
# x_axis = st.sidebar.selectbox("Choose x column", columns)
#y_axis = st.sidebar.selectbox("Выберите колонку для оси Y", columns)

#     # создаем scatter plot
# fig1 = sns.scatterplot(x=x_axis, y=y_axis, data=train)
# st.pyplot(fig1.figure)

# # создаем bar plot
# fig2 = sns.barplot(x=x_axis, y=y_axis, data=train)
# st.pyplot(fig2.figure)

# # создаем plot
# fig3 = sns.lineplot(x=x_axis, y=y_axis, data=train)
# st.pyplot(fig3.figure)
# # создаем график
plt.rcParams['font.size'] = '22'
fig, ax = plt.subplots()
ax.scatter(test['x1'].sort_values(), y_pred.round(), marker='o')
ax.scatter(test['x1'].sort_values(), test['y'], color='red', linewidth=3)
ax.grid()
ax.set_xlabel('x1')
ax.set_ylabel('y')

# устанавливаем размер графика
fig.set_size_inches(25, 10)

# отображаем график в Streamlit
st.pyplot(fig)

fig1, ax1 = plt.subplots()
ax1.scatter(test['x2'].sort_values(), y_pred.round(), marker='o')
ax1.scatter(test['x2'].sort_values(), test['y'], color='red', linewidth=3)
ax1.grid()
ax1.set_xlabel('x2')
ax1.set_ylabel('y')

# устанавливаем размер графика
fig1.set_size_inches(25, 10)

# отображаем график в Streamlit
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.scatter(test['x3'].sort_values(), y_pred.round(), marker='o')
ax2.scatter(test['x3'].sort_values(), test['y'], color='red', linewidth=3)
ax2.grid()
ax2.set_xlabel('x3')
ax2.set_ylabel('y')

# устанавливаем размер графика
fig2.set_size_inches(25, 10)

# отображаем график в Streamlit
st.pyplot(fig2)
st.write("""
### It's marvelous, isn't it?

""")



