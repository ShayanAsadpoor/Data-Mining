import pandas as pd
path = 'day.csv'
data = pd.read_csv("day.csv")
#print(data)

from sklearn import linear_model
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np

X = data[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
Y = data['cnt']


print("Linear Regresion")
model = linear_model.LinearRegression()
model.fit(X,Y)
print("Error: ")
print(np.linalg.norm(model.predict(X)-Y))
print(model.coef_)
print(model.intercept_)
print("\n")

print("Ridge Regresion")
from sklearn.linear_model import Ridge
modelridge = Ridge(alpha=1.0)
modelridge.fit(X, Y)
print("Error: ")
print(np.linalg.norm(modelridge.predict(X)-Y))
print(modelridge.coef_)
print(modelridge.intercept_)
print("\n")

print("Linear Regresion only weather")
XX = data[['temp', 'atemp', 'hum', 'windspeed']]
modelp = linear_model.LinearRegression()
modelp.fit(XX,Y)
print("Error: ")
print(np.linalg.norm(modelp.predict(XX)-Y))
print(modelp.coef_)
print(modelp.intercept_)
print("\n")

print("Decision Tree Regresion")
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0,max_depth=2)
regressor.fit(X,Y)
print("Error: ")
print(np.linalg.norm(regressor.predict(X)-Y))
print(regressor.feature_importances_)
tree.plot_tree(regressor)
plt.show()