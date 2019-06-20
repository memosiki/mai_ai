import time

import pandas as pd

df = pd.read_csv('^IXIC-recent.csv')

# преобразуем дату строкой в timestamp типа float
df['Date'] = df['Date'].map(lambda s:
                            time.mktime(time.strptime(s, "%Y-%m-%d")))
# print(df['Date'])

# если в течение дня была получена прибыль, то значение нового столбца 1
# иначе 0
profit = []
for op, cl in zip(df['Open'], df['Close']):
    profit.append(1 if cl > op else 0)
df = df.assign(Profit=profit)
# print(df['Profit'])
df.to_csv("IXIC-redacted.csv")
