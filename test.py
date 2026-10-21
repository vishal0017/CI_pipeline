import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model=joblib.load("model.pkl")

data=pd.read_csv("Iris.csv")
data=data.drop("Id", axis=1)

X=data.drop("Species",axis=1)
y=data['Species']

print(X.head())

pred=model.predict(X)
accuracy=accuracy_score(pred,y)
print(accuracy_score)

assert len(pred)==len(y)
assert accuracy>=0.50

print("All tests passed")