import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def add_to_set(res, elem):
    try:
        res += elem
    except TypeError:
        res.append(elem)


df = pd.read_csv("fake.csv")
result = []
words = df['text'].str.lower().str.strip(""",."'][""").str.split().apply(
    lambda elem: add_to_set(result, elem))

concat_text = pd.Index(result)
concat_text.value_counts().head(n=50).plot.bar()
plt.show()
