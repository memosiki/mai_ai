import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("^IXIC.csv")
# print(df.columns)
# df.plot.scatter(x='Open', y='High')
# df.plot.scatter(x='High', y='Close')
# df.plot.scatter(x='Open', y='Adj Close')

adj_price = df["Adj Close"]
print("Характеристики для скорректированной цены")
print("Среднее значение", adj_price.mean())
print("Медиана", adj_price.median())
print("Мода")
{print(val) for val in df["Adj Close"].mode().values}
print("Максимум", adj_price.max())
print("Минимум", adj_price.min())
print("Размах", adj_price.max()-adj_price.min())
print("Стандартное отклонение", adj_price.std())
print('Дисперсия', adj_price.std()**2)
plt.show()
