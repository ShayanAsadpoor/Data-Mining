import numpy as np 
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor

#Result: a; b - the most dissimilar diametric points in D
D = np.array([[0,1],[1,3],[5,0],[2,4]])

#Compute the data mean
mu = np.mean(D)

s = np.inf
for i in range(len(D)):
  for j in range(i+1,len(D)):
    temp  = D[i].T * mu + D[j].T * mu + D[i].T *D[j]
    if (temp < s).all():
      s = temp
      a = D[i:i+1]
      b = D[j:j+1]
    
print(a,b) 

#message to TA: Please make sure the path for csv file in your machine is correct.
df = pd.read_csv("day.csv")
print(df.head())

X = df.iloc[:, 2:13 ]
Y = df.iloc[:, -1 ] 

xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=.2,random_state=2)

#scikitlearn's linear regression
reg_linear = LinearRegression().fit(xtrain, ytrain)

pred_linear = reg_linear.predict(xtest)

#the corresponding coeffcient for each predictor variable
coefs_linear = pd.DataFrame(
    reg_linear.coef_,
    columns=["Coefficients"],
    index=reg_linear.feature_names_in_,
)

print(coefs_linear)

#the residual error (SSE) of the model
print("linear regression SSE: ", np.square(pred_linear - ytest).sum())

#scikitlearn's ridge regression
reg_ridge = Ridge(alpha=1.0).fit(xtrain, ytrain)

pred_ridge = reg_ridge.predict(xtest)

#the corresponding coeffcient for each predictor variable
coefs_ridge = pd.DataFrame(
    reg_ridge.coef_,
    columns=["Coefficients"],
    index=reg_ridge.feature_names_in_,
)

print(coefs_ridge)

#the residual error (SSE) of the model
print("ridge regression SSE: ", np.square(pred_ridge - ytest).sum())

#Train a linear regression using only weather predictors
# temp : Normalized temperature in Celsius. The values are divided to 41 (max)
# atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
# hum: Normalized humidity. The values are divided to 100 (max)
# windspeed: Normalized wind speed. The values are divided to 67 (max)

X = df.iloc[:, 9:13 ]
Y = df.iloc[:, -1 ] 

print(X.head())

xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=.2,random_state=2)

#linear regression using only weather predictors
reg = LinearRegression().fit(xtrain, ytrain)

pred = reg.predict(xtest)

coefs = pd.DataFrame(
    reg.coef_,
    columns=["Coefficients"],
    index=reg.feature_names_in_,
)

print(coefs)

print("SSE: ", np.square(pred - ytest).sum())

X = df.iloc[:, 2:13 ]
Y = df.iloc[:, -1 ] 

xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=.2,random_state=2)

# DecisionTreeRegressor
reg_tree = DecisionTreeRegressor(random_state=0).fit(xtrain, ytrain)

pred_tree = reg_tree.predict(xtest)

coefs_tree = pd.DataFrame(
    reg_tree.feature_importances_,
    columns=["Coefficients"],
    index=reg_tree.feature_names_in_,
)

print(coefs_tree)

print("DecisionTreeRegressor SSE: ", np.square(pred_tree - ytest).sum())