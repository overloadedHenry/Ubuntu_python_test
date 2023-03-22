import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import matplotlib as mpt
import sys
import joblib

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('./datas/boston_housing.data', sep='\s+', header=None)

# print(data)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# print(X)
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

#Feature Project
'''
PolynomialFeatures 多项式扩展
degree = 2  扩展2阶
interaction_only = False 是否只保留交互项
include_bias=True 是否需要偏置
'''

print(type(x_train))
print(x_train.shape)
print(x_test.shape)
print(x_test.iloc[0, :])
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)

poly.fit(x_train)
x_train_poly = poly.transform(x_train)
x_test_poly = poly.transform(x_test)
print("=" * 100)

# print(type(x_train_poly))
# print(x_train_poly.shape)
# print(x_test_poly.shape)
# print(x_test_poly[0])

joblib.dump(poly, "./poly.m")

# linear = LinearRegression(fit_intercept=True)
# lasso = Lasso(alpha=1.0, fit_intercept=True)
ridge = Ridge(alpha=10000, fit_intercept=True)

ridge.fit(x_train_poly, y_train)

print("*" * 100)

# print(ridge.coef_)
# print(ridge.intercept_)

y_test_hat = ridge.predict(x_test_poly)
y_train_hat = ridge.predict(x_train_poly)
print(ridge.score(x_train_poly, y_train))
print(ridge.score(x_test_poly, y_test))


plt.plot(range(len(x_test)), y_test, 'r', label=u'true')
plt.plot(range(len(x_test)), y_test_hat, 'g', label=u'predict')
plt.legend(loc='upper right')
plt.show()
