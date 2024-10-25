import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import math 
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

e=math.e
#Он может загрузить файл .csv
uploaded_file=st.sidebar.file_uploader('Загрузи CSV', type='csv')

if uploaded_file is None:
    st.write('Файл не был загружен или тип не csv')
    st.stop()

#После загрузки получает результат регрессии. (Словарик вида: имя столбца - вес столбца)
train = pd.read_csv(uploaded_file)
st.write(train.head(4))

columns=np.array(train.columns)
columns=columns[:-1]


X=train.iloc[:,:-1]
Target=np.array([train.iloc[:,-1]])
Target=Target.reshape(-1,1)


scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # добавила столбец единиц для матричных операций

class LogReg:
    def __init__(self, learning_rate, X, Target):
        self.learning_rate = learning_rate
        self.X=X
        self.n_inputs=X.shape[1]
        self.Target=Target
#        self.intercept_ = ...
       
    def grad(self,coef):
        grad2=np.array((-self.Target+self.sigmoid(self.X@coef))) #for w0
        grad2=grad2.reshape(-1,1)
        for j in range(1,self.X.shape[1]):
            grad3=self.X[:,j]*(-self.Target+self.sigmoid(self.X@coef))
            grad3=grad3.reshape(-1,1)
            grad2=np.concatenate((grad2,grad3),axis=1) #for w1 ... wn, столбец j
        return np.mean(grad2,axis=0)

    def sigmoid(self, Z):
        return (1/(1+e**(-Z)))

    def w_optimal(self,k):
        coef = np.random.uniform(-2,2,self.n_inputs)
        for i in range(k):
            w_new=coef-self.learning_rate*self.grad(coef)
            eps=np.mean(abs(self.sigmoid(self.X @ w_new) -self.sigmoid(self.X@coef)))
            coef=w_new
        return coef, eps

    def predict(self, X,k):
        coef, eps = self.w_optimal(k)
        return self.sigmoid(X@coef)

k2 = st.sidebar.number_input('Введите количество итераций для градиентного спуска', min_value=50, max_value=10000, value=2000)
learning_rate2 = st.sidebar.number_input('Введите learning rate', min_value=0.00001, max_value=0.5, value=0.01)

func=LogReg(learning_rate=learning_rate2,X=X,Target=Target)
coef, eps = func.w_optimal(k2)

a = {k:v for k, v in zip(columns, coef[1:])}
st.write(a)
st.write(f'intercept= {coef[0]}')

#А так же возможность построить scatter по любым двум фичам (цветом выделить таргет)
x=np.linspace(-1.5,1.5,100)
y=-x*coef[1]/coef[2]-coef[0]/coef[2]

fig, ax = plt.subplots()
ax.clear()
plt.scatter(X[:,1], X[:,2],c=Target, s=4)
plt.plot(x,y)
st.pyplot(fig)
