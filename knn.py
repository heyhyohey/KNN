import pandas as pd

data = pd.read_csv("stock_history.csv", encoding="ms949")
data = data.drop(['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'], 1)
print(data['basic_date'])
