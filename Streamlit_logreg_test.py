import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Функция сигмоиды
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Класс логистической регрессии с градиентным спуском
class LogReg:
    def __init__(self, learning_rate, X, Target):
        self.learning_rate = learning_rate
        self.X = X  # Матрица признаков
        self.Target = Target  # Целевая переменная
        self.n_inputs = X.shape[1]  # Количество признаков (включая bias)
   
    def grad(self, coef):
        # Вычисляем предсказания
        predictions = sigmoid(self.X @ coef)
       
        # Вычисляем ошибку (предсказания - целевые значения)
        grad_w0= predictions - self.Target.flatten()  #  для w0
        grad_w0_mean = np.mean(grad_w0)
       
        # Градиенты для остальных весов (coef[1:])
        grad_weights = np.mean(grad_w0.reshape(-1, 1) * self.X[:, 1:], axis=0)  
       
        # Возвращаем объединенный градиент
        return np.concatenate(([grad_w0_mean], grad_weights))

    def w_optimal(self, k):
        # Инициализируем коэффициенты случайными значениями
        coef = np.random.uniform(-1, 1, self.n_inputs)
       
        # Выполняем градиентный спуск
        for i in range(k):
            grad = self.grad(coef)
            coef -= self.learning_rate * grad  # Обновляем коэффициенты
       
        return coef

# Загружаем CSV файл через интерфейс Streamlit
uploaded_file = st.sidebar.file_uploader('Загрузите CSV файл', type='csv')

if uploaded_file is None:
    st.write('Файл не был загружен или тип не CSV')
    st.stop()


train = pd.read_csv(uploaded_file)
st.write(train.head(4))


X = train.iloc[:, :-1].values  # Признаки
Target = train.iloc[:, -1].values.reshape(-1, 1)  # Целевая переменная

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Добавляем столбец единиц для bias

# Получение количества итераций и learning rate от пользователя
k2 = st.sidebar.number_input('Введите количество итераций для градиентного спуска', min_value=50, max_value=10000, value=1000)
learning_rate2 = st.sidebar.number_input('Введите learning rate', min_value=0.00001, max_value=0.5, value=0.01)

# Создаем и обучаем модель логистической регрессии
func = LogReg(learning_rate=learning_rate2, X=X, Target=Target)
coef = func.w_optimal(k2)

st.write(f'Коэффициенты: {coef[1:]}')
st.write(f'Свободный член (intercept): {coef[0]}')

# Построение графика
x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y = -x * coef[1] / coef[2] - coef[0] / coef[2]

fig, ax = plt.subplots()
plt.scatter(X[:, 1], X[:, 2], c=Target, s=4)
plt.plot(x, y, color='red')
st.pyplot(fig)