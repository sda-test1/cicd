import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

df = pd.read_csv('data/otodom.csv')
df = df.dropna()
X = df[['area']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


score = model.score(X_test, y_test)
assert score > 0.7, 'Unsatisfied score'
print('Model score:', score)
dump(model, 'models/model.joblib')